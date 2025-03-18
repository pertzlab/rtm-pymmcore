from typing import TypedDict
import enum
from fov import FOV
import numpy as np
import os
from skimage.util import map_array
import lzma
import pandas as pd


class ImgType(enum.Enum):
    IMG_RAW = enum.auto()
    IMG_STIM = enum.auto()


class MetadataDict(TypedDict):
    fov: FOV
    img_type: ImgType
    last_channel: str
    stim_mask: np.array
    timestep: int
    time: int
    fname: str


def create_folders(path, folders):
    """Create all folders if they don't already exist.

    Keyword arguments:
    path -- location of main folder
    folders -- list of all subfolders
    """

    for folder in folders:
        dir_name = os.path.join(path, folder)
        try:
            os.makedirs(dir_name)
            print("Directory", dir_name, "created ")
        except FileExistsError:
            print("Directory", dir_name, "already exists")


def labels_to_particles(labels, tracks):
    """Takes in a segmentation mask with labels and replaces them with track IDs that are consistent over time."""
    # For every frame
    # labels_stack = np.array(labels_stack)
    particles = np.zeros_like(labels)
    tracks_f = tracks[(tracks["frame"] == tracks.frame.max())]
    # particle_f = np.zeros((1024,1024))
    from_label = tracks_f["label"].values
    to_particle = tracks_f["particle"].values
    particles = map_array(labels, from_label, to_particle, out=particles)
    return particles


def write_compressed_pickle(data: pd.DataFrame, filename: str) -> None:
    """Writes a compressed pickle file. Uses Lzma as compression algorithm. File extension will be added automatically (.xz)"""
    if not filename.endswith(".xz"):
        filename = filename + ".xz"
    with lzma.open(filename, "wb") as f:
        pd.to_pickle(data, f)


def read_compressed_pickle(filename: str) -> pd.DataFrame:
    """Reads a compressed pickle file that uses Lzma as compression algorithm. File extension will be added automatically (.xz)"""
    if not filename.endswith(".xz"):
        filename = filename + ".xz"
    with lzma.open(filename, "rb") as f:
        data = pd.read_pickle(f)
    return data
