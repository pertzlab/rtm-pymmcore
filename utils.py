from typing import TypedDict
import enum
from fov import FOV
import numpy as np
import os

class ImgType(enum.Enum):
    IMG_RAW = enum.auto()
    IMG_STIM = enum.auto()

class MetadataDict(TypedDict):
    fov : FOV
    img_type : ImgType
    last_channel : str 
    stim_mask : np.array
    timestep : int
    time : int
    fname : str


def create_folders(path,folders):
    """Create all folders if they don't already exist.

    Keyword arguments:
    path -- location of main folder
    folders -- list of all subfolders
    """
    
    for folder in folders:
        dir_name = path + folder
        try:
            os.makedirs(dir_name)
            print("Directory" , dir_name ,  "created ") 
        except FileExistsError:
            print("Directory" , dir_name ,  "already exists")