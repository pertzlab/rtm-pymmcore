"""Stim masks are correct by construction: the runner pulls events serially.

The MDA runner pulls the next event from ``Controller._event_stream`` only
after the previous event fully completed, including the synchronous
``frameReady`` handling that submits the frame to the pipeline. A stim
event's source frame, ``(t-1, p)`` in ``previous`` mode and ``(t, p)`` in
``current`` mode, is therefore already in the pipeline when its mask is
requested. The only remaining wait is segmentation latency.

These tests assert the pull-order invariant end to end and that a late
mask delivery still fires, with a warning. The all-off fallback path is
covered end to end in ``tests/test_pipeline_stim.py``.
"""

from __future__ import annotations

import time

import pytest

from faro.core.controller import Analyzer, Controller
from faro.core.data_structures import ImgType
from faro.stimulation.center_circle import CenterCircle
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline

N_TIMEPOINTS = 6
STIM_FRAMES = (2, 3, 4, 5)
CAMERA_DELAY_S = 0.20  # slow enough that any event read-ahead would show


def _tracker():
    return TrackerTrackpy(search_range=50, memory=3)


class SlowCircleScene(CircleScene):
    """CircleScene with a camera slow enough to expose event read-ahead."""

    def render(self, event):
        time.sleep(CAMERA_DELAY_S)
        return super().render(event)


class DelayedStimulator(CenterCircle):
    """CenterCircle whose mask takes ~1 s, arriving past the event's due time."""

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        time.sleep(1.0)
        return super().get_stim_mask(label_images, metadata, img=img, tracks=tracks)


@pytest.mark.parametrize(
    ("stim_mode", "source_offset"), [("previous", -1), ("current", 0)]
)
def test_stim_mask_requested_only_after_source_frame_submitted(
    tmp_dir, monkeypatch, stim_mode, source_offset
):
    pipeline = make_pipeline(tmp_dir, tracker=_tracker(), with_stim=True)
    mic = FakeMicroscope(SlowCircleScene(with_slm=True))
    ctrl = Controller(mic, pipeline)

    # Record each imaging frame's (t, p) at pipeline-submission time, and at
    # each stim build whether its source frame is already among them. That
    # is exactly the invariant the pull architecture guarantees.
    submitted: set[tuple[int, int]] = set()
    orig_run = Analyzer.run

    def recording_run(self, img, event):
        meta = event.metadata or {}
        if meta.get("img_type", ImgType.IMG_RAW) != ImgType.IMG_STIM:
            submitted.add((event.index.get("t", 0), event.index.get("p", 0)))
        return orig_run(self, img, event)

    monkeypatch.setattr(Analyzer, "run", recording_run)

    build_stim_slm = ctrl._build_stim_slm
    builds: list[tuple[int, bool]] = []

    def record_build(rtm_event, **kwargs):
        t = rtm_event.index.get("t", 0)
        p = rtm_event.index.get("p", 0)
        builds.append((t, (t + source_offset, p) in submitted))
        return build_stim_slm(rtm_event, **kwargs)

    ctrl._build_stim_slm = record_build

    events = make_events(N_TIMEPOINTS, stim_frames=STIM_FRAMES)
    status = ctrl.run_experiment(events, stim_mode=stim_mode, validate=False).wait()
    ctrl._analyzer.wait_idle(timeout=120)
    analyzer_errors = list(ctrl._analyzer.background_errors)
    ctrl._analyzer.shutdown(wait=True)

    assert analyzer_errors == []
    assert [t for t, _ in builds] == list(STIM_FRAMES)
    assert [t for t, in_pipeline in builds if not in_pipeline] == []
    assert status.state == "done"
    assert status.n_stim_fallbacks == 0
    assert len(mic.scene.slm_events) >= len(STIM_FRAMES)


def test_late_mask_still_fires_and_warns(tmp_dir, capsys):
    # An unscheduled stim event is due the moment its build starts, so a
    # mask that takes ~1 s to compute arrives late. It must still fire
    # with the real pattern (not the fallback), with a warning.
    pipeline = make_pipeline(
        tmp_dir, tracker=_tracker(), stimulator=DelayedStimulator()
    )
    mic = FakeMicroscope(CircleScene(with_slm=True))
    ctrl = Controller(mic, pipeline)
    events = make_events(4, stim_frames=(2,))
    status = ctrl.run_experiment(events, stim_mode="current", validate=False).wait()
    ctrl._analyzer.wait_idle(timeout=120)
    ctrl._analyzer.shutdown(wait=True)

    assert status.state == "done"
    assert status.n_stim_fallbacks == 0
    ((event_t, slm),) = mic.scene.slm_events
    assert event_t == 2
    assert slm.any()
    assert "past the event's scheduled time" in capsys.readouterr().out
