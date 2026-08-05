"""Pure-Python unit tests for Pertzlab-specific faro code.

Lives under ``tests/hardware/pertzlab/`` because its subjects are
Pertzlab-scope-only: power-property mappings, ``SKIP_WAIT_DEVICES``,
filter-turret verification, and the Moench engine's DMD/SLM handling.

These do **not** require a real scope and are **not** marked
``@pytest.mark.hardware``; they run in every test session.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from useq import MDAEvent, MDASequence

from faro.core._useq_compat import SLMImage
from faro.core.data_structures import Channel, PowerChannel
from faro.microscope.pertzlab.moench import KeepDMDAlive, MoenchMDAEngine

from tests.fake_mmc import build_core


# ===================================================================
# Power-property mapping (manual-only; unmapped power fails loud)
# ===================================================================

class TestPowerPropertyMapping:
    """Power mappings are declared on the microscope; no auto-detection."""

    def _mic(self, mapping):
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mic = PyMMCoreMicroscope()
        mic.POWER_PROPERTIES = mapping
        return mic

    def test_get_power_properties_is_manual_only(self):
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        assert mic.get_power_properties() == {"CyanStim": ("LED", "Cyan_Level")}

    def test_resolve_power_mapped(self):
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        ch = PowerChannel(config="CyanStim", exposure=10, power=25)
        assert mic.resolve_power(ch) == ("LED", "Cyan_Level", 25)

    def test_resolve_power_none_without_power(self):
        """Plain Channels and power-less PowerChannels resolve to None."""
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        assert mic.resolve_power(Channel(config="BF", exposure=10)) is None
        assert mic.resolve_power(PowerChannel(config="x", exposure=10)) is None

    def test_resolve_power_raises_when_unmapped(self):
        """A PowerChannel with power set but no mapping must fail loud."""
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        ch = PowerChannel(config="mScarlet3", exposure=10, power=10)
        with pytest.raises(ValueError, match="mScarlet3"):
            mic.resolve_power(ch)

    def test_validate_hardware_flags_unmapped_power(self):
        """validate_hardware warns + fails when a power channel has no mapping."""
        from faro.core.utils import validate_hardware

        class _StubMMC:
            def getAvailableConfigGroups(self):
                return ["TTL_ERK"]

            def getAvailableConfigs(self, group):
                return ["mScarlet3"]

            def getCameraDevice(self):
                return ""  # skip the exposure-limit block

        events = [
            SimpleNamespace(
                channels=[PowerChannel(config="mScarlet3", exposure=200, power=10)],
                stim_channels=[],
                ref_channels=[],
            )
        ]
        with pytest.warns(UserWarning, match="no power-property mapping"):
            ok = validate_hardware(events, _StubMMC(), power_properties={})
        assert ok is False


# ===================================================================
# SKIP_WAIT_DEVICES — per-microscope waitForDevice skip list
# ===================================================================


class _FakeWaitMMCore:
    """Minimal mmcore surface for _wait_for_system_excluding_xy tests."""

    def __init__(self, devices: list[str], xy_stage: str = ""):
        self._devices = devices
        self._xy_stage = xy_stage
        self.wait_calls: list[str] = []

    def getLoadedDevices(self):
        return self._devices

    def getXYStageDevice(self):
        return self._xy_stage

    def waitForDevice(self, dev: str):
        self.wait_calls.append(dev)

    def getXYPosition(self):
        return (0.0, 0.0)


class _FakeMoench:
    """Weakref-able stand-in for Moench carrying SKIP_WAIT_DEVICES."""

    def __init__(self, skip: tuple[str, ...]):
        self.SKIP_WAIT_DEVICES = skip


class TestSkipWaitDevices:
    """MoenchMDAEngine honors the microscope's SKIP_WAIT_DEVICES tuple.

    Regression guard against the 5 s-per-event Mosaic3 stuck-Busy wait
    (TODO.md #1): devices listed here must never hit waitForDevice().
    """

    def _make_engine(self, mmc, mic):
        import weakref

        from faro.microscope.pertzlab.moench import MoenchMDAEngine

        # Bypass MDAEngine.__init__ — the base class wants a real
        # CMMCorePlus, but this test only exercises the skip-filter
        # logic which reads self.mmcore and self.microscope.
        engine = MoenchMDAEngine.__new__(MoenchMDAEngine)
        engine._mmcore_ref = weakref.ref(mmc)
        engine._microscope_ref = weakref.ref(mic)
        return engine

    def test_skip_devices_bypass_wait(self):
        from useq import MDAEvent

        mmc = _FakeWaitMMCore(
            devices=["Core", "Camera", "Shutter", "Mosaic3", "XYStage"],
            xy_stage="XYStage",
        )
        mic = _FakeMoench(skip=("Mosaic3",))
        engine = self._make_engine(mmc, mic)

        engine._wait_for_system_excluding_xy(MDAEvent())

        assert "Mosaic3" not in mmc.wait_calls, "Mosaic3 should be skipped"
        assert "XYStage" not in mmc.wait_calls, "xy_stage handled separately"
        assert "Core" not in mmc.wait_calls, "Core is always skipped"
        assert "Camera" in mmc.wait_calls
        assert "Shutter" in mmc.wait_calls

    def test_missing_attribute_is_noop(self):
        """Microscopes without SKIP_WAIT_DEVICES fall through to the default."""
        from useq import MDAEvent

        mmc = _FakeWaitMMCore(
            devices=["Core", "Camera", "Mosaic3"],
            xy_stage="",
        )

        class _BareMic:
            pass

        engine = self._make_engine(mmc, _BareMic())
        engine._wait_for_system_excluding_xy(MDAEvent())

        # Without SKIP_WAIT_DEVICES, Mosaic3 is waited on as before.
        assert "Mosaic3" in mmc.wait_calls
        assert "Camera" in mmc.wait_calls


# ===================================================================
# Filter-turret verify / force-move (NikonTI "Already at position" bug)
# ===================================================================

class _Setting:
    def __init__(self, dev, prop, val):
        self._dev, self._prop, self._val = dev, prop, val

    def getDeviceLabel(self):
        return self._dev

    def getPropertyName(self):
        return self._prop

    def getPropertyValue(self):
        return self._val


class _Config:
    def __init__(self, settings):
        self._s = settings

    def size(self):
        return len(self._s)

    def getSetting(self, i):
        return self._s[i]


class _FakeFilterMMC:
    """Minimal mmcore surface for ``_verify_filter_block``.

    ``getState`` returns a *scripted* sequence so the test controls exactly
    what the read-back sees (decoupled from set* calls), which is what lets us
    assert the engine's control flow precisely.
    """

    DEVICE = "TIFilterBlock1"
    TARGET_LABEL = "cube_T"
    TARGET_STATE = 2
    N_STATES = 6

    def __init__(self, read_states, *, loaded=True, config_has_filter=True):
        self._reads = list(read_states)
        self._stuck = self._reads[-1] if self._reads else 0
        self._loaded = [self.DEVICE] if loaded else ["Camera"]
        self._config_has_filter = config_has_filter
        self.setState_calls: list[int] = []
        self.setStateLabel_calls: list[str] = []
        self.getState_calls = 0

    def getLoadedDevices(self):
        return self._loaded

    def getConfigData(self, group, config):
        settings = [_Setting("Wheel-A", "Label", "x")]
        if self._config_has_filter:
            settings.append(_Setting(self.DEVICE, "Label", self.TARGET_LABEL))
        return _Config(settings)

    def getStateFromLabel(self, device, label):
        assert label == self.TARGET_LABEL
        return self.TARGET_STATE

    def getNumberOfStates(self, device):
        return self.N_STATES

    def waitForDevice(self, device):
        pass

    def getState(self, device):
        self.getState_calls += 1
        return self._reads.pop(0) if self._reads else self._stuck

    def setState(self, device, n):
        self.setState_calls.append(n)

    def setStateLabel(self, device, label):
        self.setStateLabel_calls.append(label)


class _FilterMic:
    FILTER_VERIFY_DEVICE = "TIFilterBlock1"
    FILTER_VERIFY_PROPERTY = "Label"
    FILTER_VERIFY_MAX_CORRECTIONS = 3
    FILTER_VERIFY_RAISE_ON_FAILURE = False


class TestFilterBlockVerify:
    """MoenchMDAEngine confirms the cube turret landed and force-moves on a miss.

    Guards the silent "Already at position; not moving" skip in the closed
    NikonTI adapter (see ``nikonti-re/FINDINGS.md``).
    """

    def _make_engine(self, mmc, mic):
        import weakref

        from faro.microscope.pertzlab.moench import MoenchMDAEngine

        engine = MoenchMDAEngine.__new__(MoenchMDAEngine)
        engine._mmcore_ref = weakref.ref(mmc)
        engine._microscope_ref = weakref.ref(mic)
        return engine

    def test_fast_path_no_extra_move(self):
        # Turret already reads the target -> no rotation, no correction.
        mmc = _FakeFilterMMC(read_states=[_FakeFilterMMC.TARGET_STATE])
        mic = _FilterMic()  # keep a strong ref; engine holds it weakly
        engine = self._make_engine(mmc, mic)

        engine._verify_filter_block("TTL_ERK", "mScarlet3")

        assert mmc.setState_calls == []
        assert mmc.setStateLabel_calls == []

    def test_recovers_on_detected_mismatch(self):
        # First read wrong (suppressed move), recovers after one force-move.
        mmc = _FakeFilterMMC(read_states=[0, _FakeFilterMMC.TARGET_STATE])
        mic = _FilterMic()  # keep a strong ref; engine holds it weakly
        engine = self._make_engine(mmc, mic)

        engine._verify_filter_block("TTL_ERK", "mScarlet3")

        # neighbour = (2 + 1) % 6 = 3, then the real target label.
        assert mmc.setState_calls == [3]
        assert mmc.setStateLabel_calls == ["cube_T"]

    def test_persistent_failure_logs_but_does_not_raise(self):
        # Always wrong: exhaust corrections, log loudly, do not raise (default).
        mmc = _FakeFilterMMC(read_states=[0])  # stuck at 0 forever
        mic = _FilterMic()  # keep a strong ref; engine holds it weakly
        engine = self._make_engine(mmc, mic)

        engine._verify_filter_block("TTL_ERK", "mScarlet3")

        assert len(mmc.setStateLabel_calls) == 3  # MAX_CORRECTIONS attempts
        assert mmc.setState_calls == [3, 3, 3]

    def test_persistent_failure_raises_when_flagged(self):
        mmc = _FakeFilterMMC(read_states=[0])
        mic = _FilterMic()
        mic.FILTER_VERIFY_RAISE_ON_FAILURE = True
        engine = self._make_engine(mmc, mic)

        with pytest.raises(RuntimeError, match="WRONG cube"):
            engine._verify_filter_block("TTL_ERK", "mScarlet3")

    def test_channel_without_turret_is_noop(self):
        # A channel whose preset doesn't drive the turret is left alone.
        mmc = _FakeFilterMMC(read_states=[0], config_has_filter=False)
        mic = _FilterMic()  # keep a strong ref; engine holds it weakly
        engine = self._make_engine(mmc, mic)

        engine._verify_filter_block("Binning", "2x2")

        assert mmc.getState_calls == 0
        assert mmc.setStateLabel_calls == []

    def test_device_not_loaded_is_noop(self):
        mmc = _FakeFilterMMC(read_states=[0], loaded=False)
        mic = _FilterMic()  # keep a strong ref; engine holds it weakly
        engine = self._make_engine(mmc, mic)

        engine._verify_filter_block("TTL_ERK", "mScarlet3")

        assert mmc.getState_calls == 0
        assert mmc.setStateLabel_calls == []

# ===================================================================
# DMD/SLM uploads, live-stop, and keep-alive lifecycle
# ===================================================================


class _SLMScene:
    """Minimal scene declaring a camera and an SLM; never renders."""

    image_height = image_width = 64
    channels = ("phase-contrast",)
    slm_name = "SLM"
    slm_shape = (64, 64)

    def render(self, event):
        return np.zeros((self.image_height, self.image_width), dtype=np.uint16)


@pytest.fixture()
def slm_core():
    return build_core(_SLMScene())


def _record_slm_calls(core):
    """Replace the core's two SLM upload paths with recorders."""
    calls = []
    core.setSLMImage = lambda label, image: calls.append(("setSLMImage", image))
    core.setSLMPixelsTo = lambda *args: calls.append(("setSLMPixelsTo", args))
    return calls


class TestScalarSLMImageExpansion:
    """Every SLM command reaches the core as an array, never as a scalar.

    A scalar all-on/all-off goes out via ``setSLMPixelsTo``, which some DMDs
    ignore without raising, leaving the last pattern on the mirrors.
    """

    @pytest.mark.parametrize(
        ("data", "expected_value"), [(True, 255), (False, 0)], ids=["all-on", "all-off"]
    )
    def test_scalar_routed_through_set_slm_image(self, slm_core, data, expected_value):
        engine = MoenchMDAEngine(slm_core)
        calls = _record_slm_calls(slm_core)

        engine._set_event_slm_image(
            MDAEvent(slm_image=SLMImage(data=data, device="SLM"))
        )

        assert [name for name, _ in calls] == ["setSLMImage"]
        image = calls[0][1]
        assert image.shape == _SLMScene.slm_shape
        assert image.dtype == np.uint8
        assert (image == expected_value).all()

    def test_array_is_passed_through_unchanged(self, slm_core):
        engine = MoenchMDAEngine(slm_core)
        calls = _record_slm_calls(slm_core)
        mask = np.zeros(_SLMScene.slm_shape, dtype=np.uint8)
        mask[10:20, 10:20] = 255

        engine._set_event_slm_image(
            MDAEvent(slm_image=SLMImage(data=mask, device="SLM"))
        )

        assert [name for name, _ in calls] == ["setSLMImage"]
        assert np.array_equal(calls[0][1], mask)


class TestSetupSequenceStopsLive:
    """Every MDA stops live acquisition first, whether or not it is running.

    A viewer can keep a live timer armed after the stream itself has stopped,
    and only the ``sequenceAcquisitionStopped`` signal clears it.
    """

    def test_stops_live_when_nothing_is_running(self, slm_core):
        stopped = []
        slm_core.events.sequenceAcquisitionStopped.connect(
            lambda *args: stopped.append(args)
        )
        engine = MoenchMDAEngine(slm_core)

        assert not slm_core.isSequenceRunning()
        engine.setup_sequence(MDASequence())

        assert stopped, "sequenceAcquisitionStopped must fire even when idle"


class TestKeepDMDAliveLifecycle:
    """The keep-alive thread runs while live acquisition does, and not longer.

    It refreshes the DMD so the mirrors do not park during live view. An MDA
    drives the DMD itself, so the thread must be off for the whole run.
    """

    def _keep_alive(self, core):
        displays = []
        dmd = SimpleNamespace(display_livemode=lambda: displays.append(1))
        return KeepDMDAlive(core, dmd), displays

    def test_repeated_run_does_not_spawn_a_second_thread(self, slm_core):
        keep_alive, _ = self._keep_alive(slm_core)
        try:
            keep_alive.run()
            first = keep_alive.thread
            keep_alive.run()
            assert keep_alive.thread is first
        finally:
            keep_alive.stop()

    def test_stop_while_idle_leaves_the_slm_alone(self, slm_core):
        keep_alive, _ = self._keep_alive(slm_core)
        displayed = []
        slm_core.displaySLMImage = lambda *args: displayed.append(args)

        keep_alive.stop()

        assert displayed == []

    def test_live_signals_drive_the_thread(self, slm_core):
        """Live start runs the thread, live stop stops it."""
        keep_alive, _ = self._keep_alive(slm_core)
        slm_core.events.continuousSequenceAcquisitionStarted.connect(keep_alive.run)
        slm_core.events.sequenceAcquisitionStopped.connect(keep_alive.stop)
        try:
            slm_core.events.continuousSequenceAcquisitionStarted.emit("Camera")
            assert keep_alive.thread is not None

            slm_core.events.sequenceAcquisitionStopped.emit("Camera")
            assert keep_alive.thread is None
        finally:
            keep_alive.stop()
