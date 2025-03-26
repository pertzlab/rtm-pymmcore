import queue


class FOV:
    def __init__(self, pos, index, name, path, metadata, treatment):
        self.pos = pos  # stage position
        self.index = index  # unique index
        self.name = (
            name  # name comming from the napari interface (Pos col, but can be edited)
        )
        self.metadata = metadata  # dict with metadata
        self.treatment = treatment  # dict to store data required for experiment
        self.light_mask = None
        self.path = path  # folder with experiment
        self.stim_params = (
            {}
        )  # dict with the parameters the stimulator will unpack in the segment method
        self.stim_mask_queue = queue.SimpleQueue()
        self.tracks_queue = queue.SimpleQueue()
        self.start_time = None
        self.mda_sequence = None  # mda sequence object used for timing
        self.tracks = None  # tracks dataframe
        self.linker = None  # Linker object that will be stored from trackpy

        self.last_stim_mask = queue.LifoQueue(maxsize=1)
