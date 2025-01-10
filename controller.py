from pymmcore_plus import CMMCorePlus
#from MDAEngine_DMD import MDAEngine_DMD
#from pymmcore_plus.mda import MDAEngine
from utils import ImgType, MetadataDict
from add_frame import store_img, ImageProcessingPipeline
#import time

from fov import FOV
import useq
from useq import MDAEvent
from queue import Queue
import numpy as np
from dmd import DMD
import threading
import pandas as pd


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
            # if self.pipeline is not None:
            #     thread = threading.Thread(target=self.pipeline.run, args=(img, event))
            #     thread.start()
            print("Store raw image")
            self.pipeline.run(img,event)
            store_img(img,metadata,'raw')

        if img_type == ImgType.IMG_STIM:
            #stim image, store
            store_img(img,metadata,'stim')
            print("Store stim image")
        #TODO remove print statement and return something useful
        return {"result": "STOP"}


class Controller:
    STOP_EVENT = object()

    def __init__(self, analyzer: Analyzer, mmc: CMMCorePlus, queue: Queue, dmd=None):
        self._queue = queue  # queue of MDAEvents
        self._analyzer = analyzer  # analyzer object
        self._results: dict = {}  # results of analysis
        self._mmc = mmc
        self._frame_buffer = [] # buffer to hold the frames until one sequence is complete
        self._dmd = dmd
        mmc.mda.events.frameReady.connect(self._on_frame_ready)


    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        # Analyze the image
        self._frame_buffer.append(img)
        
        # check if it's the last acquisition for this MDAsequence
        if event.metadata['last_channel']:
            frame_complete = np.stack(self._frame_buffer, axis=-1)
            #move new axis to the first position
            frame_complete = np.moveaxis(frame_complete, -1, 0)

            self._frame_buffer = []
            #self._results = self._analyzer.run(frame_complete,event.metadata)
            self._results = self._analyzer.run(frame_complete,event) 
        

    def stop_run(self): 
        self._queue.put(self.STOP_EVENT)
        self._mmc.mda.cancel()
        
    def is_running(self):
        return self._queue.qsize() > 0

    def run(self, df_acquire:pd.DataFrame):
        try: 
            current_channelgroup = self._mmc.getChannelGroup()

            #convert queue to an iterable
            queue_sequence = iter(self._queue.get, self.STOP_EVENT)
            self._mmc.run_mda(queue_sequence)

            for timestep in df_acquire['timestep'].unique():
            # extract the lines with the current timestep from the DF
                current_timestep_df = df_acquire[df_acquire['timestep'] == timestep]

                for index, row in current_timestep_df.iterrows():
                    fov : FOV = row['fov_object']
                    timestep = row['timestep']
                    stim = row['stim']
                    channels = row['channels']
                    channels_exposure = row['channels_exposure']
                    channel_stim = row['channel_stim']
                    channel_stim_exposure = row['channel_stim_exposure']

                    metadata_dict = dict(row)
                    metadata_dict['img_type']= ImgType.IMG_RAW
                    metadata_dict['last_channel']= channels[-1]
                                                
                    if self._dmd != None:
                        metadata_dict['stim_mask'] = self._dmd.sample_mask_on
                        
                    ### Capture the raw image without DMD illumination
                    for i,channel in enumerate(channels):
                        last_channel:bool = i == len(channels)-1
                        metadata_dict['last_channel'] = last_channel
                        metadata_dict['channel'] = channel

                        acquisition_event = useq.MDAEvent(
                                index= {"t": timestep}, # the index of the event in the sequence
                                channel = {"config":channel, "group":current_channelgroup}, # the channel presets we want to acquire
                                metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                                x_pos = fov.pos[0], # only one pos for all channels
                                y_pos = fov.pos[1],
                                # sequence = fov.mda_sequence,
                                min_start_time = float(row['time']),
                                exposure=channels_exposure[i]
                            )
                        
                        #add the event to the acquisition queue
                        self._queue.put(acquisition_event)

                    if stim:
                        ### Stimulate using the DMD if stim is True
                        stim_mask = fov.stim_mask_queue.get(timeout=10) #wait max 10s for mask
                        #affine transform the mask to the DMD coordinates
                        if self._dmd != None:
                            stim_mask = self._dmd.affine_transform(stim_mask)

                        if len(channel_stim) == 1:
                            channel_stim = channel_stim[0]
                            channel_stim_exposure = channel_stim_exposure[0]
                        ### expose the image
                        metadata_dict['img_type'] = ImgType.IMG_STIM #change the img_type and channels, rest stays the same
                        metadata_dict['last_channel'] = True
                        metadata_dict['channel'] = channel_stim      

                        stimulation_event = useq.MDAEvent(
                            index= {"t": timestep}, # the index of the event in the sequence
                            channel = {"config":channel_stim, "group":current_channelgroup}, # the channel presets we want to acquire
                            metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                            x_pos = fov.pos[0], # only one pos for all channels
                            y_pos = fov.pos[1],
                            exposure = channel_stim_exposure, 
                            min_start_time=float(row['time'])
                        )
                        self._queue.put(stimulation_event)   

            # # Reached end of acquisition DF
            for fov in df_acquire['fov_object'].unique():
                fov.tracks = fov.tracks_queue.get(timeout=10) #wait max 10s for tracks

            # # Put the stop event in the queue
        finally: 
            self._queue.put(self.STOP_EVENT)