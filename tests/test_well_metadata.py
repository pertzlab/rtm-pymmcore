"""Well IDs (plate_row/plate_col) flow from imported positions to metadata."""

from __future__ import annotations

from types import SimpleNamespace

from faro.core.conversion import df_to_events
from faro.core.data_structures import Channel
from faro.core.utils import generate_df_acquire_simple, generate_fov_positions_from_list

CHANNELS = [Channel(config="phase-contrast", exposure=50)]
MIC = SimpleNamespace(USE_ONLY_PFS=False)


def _df(positions, n_frames=1):
    fovs = generate_fov_positions_from_list(MIC, positions)
    return generate_df_acquire_simple(
        fovs, n_frames=n_frames, time_between_timesteps=10, channels=CHANNELS
    )


def test_wells_flow_from_positions_to_frame_metadata():
    # Position.model_dump() dicts as _get_mda_from_viewer returns them.
    df = _df(
        [{"x": 0.0, "y": 0.0, "z": 1.0, "name": "Pos0", "plate_row": 1, "plate_col": 6}]
    )
    assert (df["plate_row"] == 1).all()
    assert (df["plate_col"] == 6).all()

    # base_meta of to_mda_events is what lands on every track row.
    for ev in df_to_events(df)[0].to_mda_events():
        assert ev.metadata["plate_row"] == 1
        assert ev.metadata["plate_col"] == 6


def test_no_wells_no_columns():
    df = _df([{"x": 0.0, "y": 0.0, "z": 1.0, "name": "Pos0"}])
    assert "plate_row" not in df.columns
    assert "plate_col" not in df.columns
    assert "plate_row" not in df_to_events(df)[0].metadata


def test_mixed_wells_leave_nan_for_unassigned():
    df = _df(
        [
            {"x": 0.0, "y": 0.0, "z": 1.0, "name": "a", "plate_row": 0, "plate_col": 0},
            {"x": 1.0, "y": 1.0, "z": 1.0, "name": "b"},
        ]
    )
    assert (df.loc[df["fov"] == 0, "plate_row"] == 0).all()
    assert df.loc[df["fov"] == 1, "plate_row"].isna().all()
