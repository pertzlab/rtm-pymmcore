from pymmcore_plus import CMMCorePlus
#from MDAEngine_DMD import MDAEngine_DMD
#from pymmcore_plus.mda import MDAEngine
from utils import ImgType, MetadataDict
from add_frame import store_img, ImageProcessingPipeline
#import time
import threading
from useq._mda_event import SLMImage
from useq import PropertyTuple, HardwareAutofocus

from fov import FOV
import useq
from useq import MDAEvent
from queue import Queue
import numpy as np
from dmd import DMD
import threading
import pandas as pd
import matplotlib.pyplot as plt
import time

"""Simple simulator demonstrating event-driven acquisitions with pymmcore-plus.
    Queue pattern from Kyle M. Douglass: https://gist.github.com/kmdouglass/d15a0410d54d6b12df8614b404d9b751
    Or find it here: https://pymmcore-plus.github.io/pymmcore-plus/guides/event_driven_acquisition/"""

class Analyzer:
    """When a new image is acquired, decide what to do here. Segment, get stim mask, just store"""

    def __init__(self, pipeline: ImageProcessingPipeline = None):
        self.pipeline = pipeline

    
    def run(self, img:np.array,event:MDAEvent) -> dict:

        metadata : MetadataDict = event.metadata
        fov = metadata['fov']
        img_type = metadata['img_type']

        if img_type == ImgType.IMG_RAW:
            #raw image, send to pipeline and store
            # TODO: back to normal
            if self.pipeline is not None:
                thread = threading.Thread(target=self.pipeline.run, args=(img, event))
                thread.start()
            store_img(img,metadata,'raw')

        if img_type == ImgType.IMG_STIM:
            #stim image, store
            store_img(img,metadata,'stim')
        #TODO remove print statement and return something useful
        return {"result": "STOP"}


class Controller:
    STOP_EVENT = object()

    def __init__(self, analyzer: Analyzer, mmc, queue, dmd=None):
        self._queue = queue  # queue of MDAEvents
        self._analyzer = analyzer  # analyzer object
        self._results: dict = {}  # results of analysis
        self._current_group = mmc.getChannelGroup()
        self._frame_buffer = [] # buffer to hold the frames until one sequence is complete
        self._dmd = dmd
        self._mmc = mmc
        self._mmc.mda.events.frameReady.disconnect()
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
    

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        # Analyze the image+        
        self._frame_buffer.append(img)
        # check if it's the last acquisition for this MDAsequence
        if event.metadata['last_channel']:
            frame_complete = np.stack(self._frame_buffer, axis=-1)
            #move new axis to the first position
            frame_complete = np.moveaxis(frame_complete, -1, 0)

            self._frame_buffer = []
            self._results = self._analyzer.run(frame_complete,event)
        


    def stop_run(self): 
        self._queue.put(self.STOP_EVENT)
        self._mmc.mda.cancel()
        
    def is_running(self):
        return self._queue.qsize() > 0
    
    def run(self, df_acquire:pd.DataFrame):
        queue_sequence = iter(self._queue.get, self.STOP_EVENT)
        self._mmc.run_mda(queue_sequence)
        try: 
            for exp_time in df_acquire['time'].unique():
                # extract the lines with the current timestep from the DF
                current_time_df = df_acquire[df_acquire['time'] == exp_time]
                for index, row in current_time_df.iterrows():
                    fov : FOV = row['fov_object']
                    timestep = row['timestep']
                    treatment = row['treatment']
                    stim = timestep in treatment["stim_timestep"]
                    stim_exposure = treatment["stim_exposure"]
                    stim_profile = treatment["stim_profile"]

                    channels = row['channels']
                    channels_exposure = row['channels_exposure']


                    acquisition_event = useq.MDAEvent(
                        index= {"t": timestep, "c": 0, "p": fov.index}, # the index of the event in the sequence
                        x_pos = fov.pos[0], # only one pos for all channels
                        y_pos = fov.pos[1],
                        min_start_time = float(row['time']),
                        action=HardwareAutofocus()
                    )
                    self._queue.put(acquisition_event)
                    
                    if timestep > 0:
                        fov.tracks = fov.tracks_queue.get(block=True) #wait max 10s for tracks

                    metadata_dict = dict(row)
                    metadata_dict['img_type']= ImgType.IMG_RAW
                    metadata_dict['last_channel']= channels[-1]

                    ### Capture the raw image without DMD illumination
                    for i, channel in enumerate(channels):
                        last_channel: bool = i == len(channels) - 1
                        metadata_dict['last_channel'] = last_channel
                        metadata_dict['channel'] = channel

                        acquisition_event = useq.MDAEvent(
                                index= {"t": timestep, "c": i, "p": fov.index}, # the index of the event in the sequence
                                channel = {"config":channel, "group":self._current_group}, # the channel presets we want to acquire
                                metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                                x_pos = fov.pos[0], # only one pos for all channels
                                y_pos = fov.pos[1],
                                min_start_time = float(row['time']),
                                exposure=channels_exposure[i]
                            )
                        #add the event to the acquisition queue
                        self._queue.put(acquisition_event)
                    if stim:
                        metadata_dict['img_type'] = ImgType.IMG_STIM #change the img_type and channels, rest stays the same
                        metadata_dict['last_channel'] = True
                        metadata_dict['channel'] = stim_profile["channel"]   
                        zero_stim_prop = PropertyTuple(stim_profile["device_name"], stim_profile["property_name"], 0)

                        if stim_exposure == 0: # not really a good solution, need to improve
                            # no stimulation, just do an image: 
                            stimulation_event = useq.MDAEvent(
                                index= {"t": timestep, "p": fov.index}, # the index of the event in the sequence
                                channel = {"config":stim_profile["channel"], "group":self._current_group}, # the channel presets we want to acquire
                                metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                                x_pos = row['fov_object'].pos[0], # only one pos for all channels
                                y_pos = row['fov_object'].pos[1],
                                exposure = 1, 
                                min_start_time=float(row['time']),
                                properties=[zero_stim_prop]
                            )
                        else:                       
                            if self._dmd != None:
                                stim_mask = fov.stim_mask_queue.get(block=True) #TODO: Not really a good idea, but timeout is also not good, as 
                                # the queue fills up already much in advance of the actual acquisition for optofgfr experiments without constant stimming. 
                                # best would be to either slow down the iteration through the dataframe, or give error masks, or something else
                                if np.all(stim_mask == 1):
                                    stim_mask = True
                                else: 
                                    stim_mask = self._dmd.affine_transform(stim_mask)

                                stimulation_event = useq.MDAEvent(
                                    index= {"t": timestep, "p": fov.index}, # the index of the event in the sequence
                                    channel = {"config":stim_profile["channel"], "group":self._current_group}, # the channel presets we want to acquire
                                    metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                                    x_pos = row['fov_object'].pos[0], # only one pos for all channels
                                    y_pos = row['fov_object'].pos[1],
                                    exposure = stim_exposure, 
                                    min_start_time=float(row['time']), 
                                    slm_image=SLMImage(data=stim_mask),
                                    properties=[PropertyTuple(stim_profile["device_name"], stim_profile["property_name"], stim_profile["power"])]
                                )
                            else: 
                                stimulation_event = useq.MDAEvent(
                                    index= {"t": timestep, "p": fov.index}, # the index of the event in the sequence
                                    channel = {"config":stim_profile["channel"], "group":self._current_group}, # the channel presets we want to acquire
                                    metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                                    x_pos = row['fov_object'].pos[0], # only one pos for all channels
                                    y_pos = row['fov_object'].pos[1],
                                    exposure = stim_exposure, 
                                    min_start_time=float(row['time']), 
                                    properties=[PropertyTuple(stim_profile["device_name"], stim_profile["property_name"], stim_profile["power"])]
                                )
                        self._queue.put(stimulation_event)

                          
        finally: 
                # # Put the stop event in the queue
            self._queue.put(self.STOP_EVENT)
            while self._queue.qsize() > 0:
                time.sleep(1)