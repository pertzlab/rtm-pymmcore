from pymmcore_plus import CMMCorePlus
from fov import FOV
import useq
from useq import MDAEvent, MDASequence
from queue import Queue
import numpy as np
from matplotlib import pyplot as plt
from napari import Viewer
import numpy as np
from queue import Queue
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent
import useq
from useq._channel import Channel
from MDAEngine_DMD import MDAEngine_DMD
from controller import Controller, Analyzer
import pandas as pd
import random
import random
from utils import ImgType, MetadataDict, create_folders
from stimulation import StimExtraParameters
from dmd import DMD
from tracking import TrackerTrackpy, TrackerNoTracking

path = '/tmp/'
pos_list = [(2,2)]

create_folders(path,['raw','stim','labels','labels_rings','stim_mask','stim','tracks'])
fovs = []

for i,pos in enumerate(pos_list):
    print(pos)
    fov = FOV(pos=pos,
              index =i,
              path=path,
              metadata={},
              properties={'stim_property': 'top'},
              )
    fovs.append(fov)


df_acquire = pd.DataFrame(columns=['fov', 'timestep', 'time', 'treatment', 'acquired','stim', 'channels', 'channels_stim'])

time_between_frames = 1 #time in seconds between frames
#stim_timesteps = [2,3]  # list of timesteps
stim_timesteps= [1,2]  # list of timesteps

timestep = range(3)  # 0-20
treatments = ['stim_top', 'stim_mid',]  # list of treatments
fovs:list[FOV] = fovs
#channels = [['DAPI','FITC']]
channels = [[
    {"config": "DAPI", "exposure": 10},
    {"config": "FITC", "exposure": 10}
    ]]

#channels_stim = [[{'config': 'Cy5', 'exposure': 10}]] ## add extra [brackets] as it's unpacked when adding to the DF
channel_stim = [{'config': 'Cy5', 'exposure': 10}]

ImgType.IMG_RAW

#TODO change the SEQUENCE in multi channels
# Loop over the FOVs and randomly assign one of the treatments to it
treatments_shuffled = treatments.copy()
random.shuffle(treatments_shuffled)
delay = time_between_frames/len(fovs) #spread out the FOVs in time
for fov in fovs:
    treatment = treatments_shuffled[fov.index % len(treatments_shuffled)]
    sequence = MDASequence()
    for timestep in timestep:
        new_row = {'fov_object': fov,
                    'fov':fov.index,
                    'timestep': timestep,
                    'time': timestep*time_between_frames,
                    'time_experiment': timestep*time_between_frames + fov.index*delay + delay,
                    'treatment': treatment,
                    'acquired': False,
                    'stim': False,
                    'channels': channels,
                    'channel_stim' : channel_stim,
                    'fname' : f'{str(fov.index).zfill(3)}_{str(timestep).zfill(5)}',
                    
                    }
        df_acquire = pd.concat([df_acquire, pd.DataFrame(new_row, index=[0])])

for timestep in stim_timesteps:
    df_acquire.loc[df_acquire['timestep'] == timestep, 'stim'] = True
df_acquire = df_acquire.sort_values(by=['timestep', 'fov'])


from add_frame import ImageProcessingPipeline
from segmentation import SegmentatorStardist
from pymmcore_plus.mda import MDAEngine
from stimulation import StimExtraParameters
from controller import Analyzer

segmentator = SegmentatorStardist('2D_versatile_fluo')
stimulator = StimExtraParameters()
tracker = TrackerTrackpy()
tracker = TrackerNoTracking()
pipeline = ImageProcessingPipeline(segmentator,stimulator,tracker,segmentation_channel=0)
analyzer = Analyzer(pipeline)

mmc = CMMCorePlus()

mmc.loadSystemConfiguration()

#setup camera
mmc.initializeDevice("Camera");
mmc.setCameraDevice("Camera");

# Register the custom engine with the runner
dmd = DMD(mmc, test_mode = True)
mda_engine_dmd = MDAEngine_DMD(dmd)

mmc.mda.set_engine(mda_engine_dmd)

queue = Queue()
controller = Controller(analyzer, mmc, queue)

# Start the acquisition
controller.run(df_acquire)
#controller.run_test()
