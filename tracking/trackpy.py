import numpy as np
import pandas as pd
from .base_tracker import Tracker
import trackpy


class TrackerTrackpy(Tracker):
    def __init__(self, search_range=10, memory=3, adaptive_stop=3, adaptive_step=1):
        super().__init__()
        self.search_range = search_range
        self.memory = memory
        self.adaptive_stop = adaptive_stop
        self.adaptive_step = adaptive_step

    def track_cells(
        self, df_old: pd.DataFrame, df_new: pd.DataFrame, metadata
    ) -> pd.DataFrame:
        """Track cells in a dataframe using trackpy library.
        Args:
            dataframe: dataframe with columns 'x', 'y', 'label'"""

        required_columns = ["x", "y", "label"]
        missing_columns = [col for col in required_columns if col not in df_new.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        fov = metadata["fov_object"]
        df_new["frame"] = metadata["timestep"]
        df_new["time"] = metadata["time"]
        df_new["fname"] = metadata["fname"]

        coordinates = np.array(
            df_new[["x", "y"]]
        )  # Convert the df to an array of shape (shape: N, ndim) for trackpy

        if metadata["timestep"] == 0:  # or table_old == None: this is the first frame
            fov.linker = trackpy.linking.Linker(
                search_range=self.search_range,
                memory=self.memory,
                adaptive_stop=self.adaptive_stop,
                adaptive_step=self.adaptive_step,
            )

            fov.linker.init_level(
                coordinates, metadata["timestep"]
            )  # extract positions and convert to horizontal list
            df_new["particle"] = fov.linker.particle_ids
            df_tracked = df_new

        else:
            # this is not the first frame
            fov.linker.next_level(
                coordinates, metadata["timestep"]
            )  # extract positions and convert to horizontal list
            df_new["particle"] = fov.linker.particle_ids
            df_tracked = pd.concat([df_old, df_new])

        # this is against a in trackpy, where the same ID gets assigned twice in one frame
        df_tracked = df_tracked.drop_duplicates(subset=["particle", "frame"])
        df_tracked = df_tracked.reset_index(drop=True)
        return df_tracked
