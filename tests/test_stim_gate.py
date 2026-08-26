"""The feed loop waits for a stim's source frame before building its mask.

The feed loop runs a few events ahead of the camera. Building a stim's SLM
blocks on ``get_stim_mask``, so without a gate it asks the pipeline for the
mask of a frame that has not been shot yet and burns the whole mask timeout
before falling back to an all-off mask.

The source frame is ``(t-1, p)`` in ``previous`` mode and ``(t, p)`` in
``current`` mode, where the imaging events are queued from the same feed-loop
iteration as the stim that follows them.
"""

from __future__ import annotations

import time

import pytest

from faro.core.controller import Controller
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CircleScene,
    assert_no_background_errors,
    make_events,
    make_pipeline,
)

N_TIMEPOINTS = 6
STIM_FRAMES = (2, 3, 4, 5)
CAMERA_DELAY_S = 0.20  # slow enough that acquisition lags the feed loop


class SlowCircleScene(CircleScene):
    """CircleScene whose camera takes long enough to fall behind the feed loop."""

    def render(self, event):
        time.sleep(CAMERA_DELAY_S)
        return super().render(event)


@pytest.mark.parametrize(
    ("stim_mode", "source_offset"), [("previous", -1), ("current", 0)]
)
def test_stim_slm_built_only_after_source_frame_acquired(
    tmp_dir, stim_mode, source_offset
):
    pipeline = make_pipeline(
        tmp_dir, tracker=TrackerTrackpy(search_range=50, memory=3), with_stim=True
    )
    mic = FakeMicroscope(SlowCircleScene(with_slm=True))
    ctrl = Controller(mic, pipeline)

    # Record whether the source frame was already acquired at each stim-SLM
    # build, which is exactly what the gate has to guarantee.
    build_stim_slm = ctrl._build_stim_slm
    builds: list[tuple[int, bool]] = []

    def record_build(rtm_event, **kwargs):
        t = rtm_event.index.get("t", 0)
        p = rtm_event.index.get("p", 0)
        with ctrl._acquired_lock:
            builds.append((t, (t + source_offset, p) in ctrl._acquired_frames))
        return build_stim_slm(rtm_event, **kwargs)

    ctrl._build_stim_slm = record_build

    events = make_events(N_TIMEPOINTS, stim_frames=STIM_FRAMES)
    ctrl.run_experiment(events, stim_mode=stim_mode, validate=False).wait()
    ctrl._analyzer.wait_idle(timeout=120)
    ctrl._analyzer.shutdown(wait=True)

    assert_no_background_errors(ctrl)
    assert [t for t, _ in builds] == list(STIM_FRAMES)
    assert [t for t, acquired in builds if not acquired] == []
    assert len(mic.scene.slm_events) >= len(STIM_FRAMES)
