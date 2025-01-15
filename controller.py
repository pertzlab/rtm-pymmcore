from pymmcore_plus import CMMCorePlus
#from MDAEngine_DMD import MDAEngine_DMD
#from pymmcore_plus.mda import MDAEngine
from utils import ImgType, MetadataDict
from add_frame import store_img, ImageProcessingPipeline
#import time
import threading
from useq._mda_event import SLMImage
from useq import PropertyTuple

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
        print("vor raw")

        if img_type == ImgType.IMG_RAW:
            #raw image, send to pipeline and store
            # TODO: back to normal
            if self.pipeline is not None:
                thread = threading.Thread(target=self.pipeline.run, args=(img, event))
                thread.start()
            print("Store raw image")
            # self.pipeline.run(img,event)
            store_img(img,metadata,'raw')

        if img_type == ImgType.IMG_STIM:
            #stim image, store
            store_img(img,metadata,'stim')
            print("Store stim image")
        #TODO remove print statement and return something useful
        return {"result": "STOP"}


class Controller:

    def __init__(self, analyzer: Analyzer, mmc: CMMCorePlus, queue: Queue, stop_event, dmd=None):
        self._queue = queue  # queue of MDAEvents
        self._analyzer = analyzer  # analyzer object
        self._results: dict = {}  # results of analysis
        self._mmc = mmc
        self._frame_buffer = [] # buffer to hold the frames until one sequence is complete
        self._dmd = dmd
        # mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self.STOP_EVENT = stop_event


    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        # Analyze the image+        
        print("New frame ready")
        self._frame_buffer.append(img)
        print(f"Frame buffer length: {len(self._frame_buffer)}")
        print(event.metadata)
        plt.imshow(img)
        plt.show()
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
    
    def _prepare_stim_event(self, row, current_channelgroup):
        fov = row['fov_object']
        stim_mask = fov.stim_mask_queue.get(timeout=20)  # wait max 10s for mask
        if self._dmd is not None:
            stim_mask = self._dmd.affine_transform(stim_mask)
        if np.all(stim_mask == 1):
            stim_mask = True
        metadata_dict = dict(row)
        metadata_dict['img_type'] = ImgType.IMG_STIM
        metadata_dict['last_channel'] = True
        metadata_dict['channel'] = row['stim_profile']['channel']
        stimulation_event = useq.MDAEvent(
            index={"t": row['timestep']},
            channel={"config": row['stim_profile']['channel'], "group": current_channelgroup},
            metadata=metadata_dict,
            x_pos=row['fov_object'].pos[0],
            y_pos=row['fov_object'].pos[1],
            exposure=row['stim_exposure'],
            min_start_time=float(row['time']),
            slm_image=SLMImage(data=stim_mask),
        )
        self._queue.put(stimulation_event)


    def run(self, df_acquire:pd.DataFrame):
        try: 
            current_channelgroup = self._mmc.getChannelGroup()
            for timestep in df_acquire['timestep'].unique():
                # extract the lines with the current timestep from the DF
                    current_timestep_df = df_acquire[df_acquire['timestep'] == timestep]

                    for index, row in current_timestep_df.iterrows():
                        print(row)
                        print("neue zeile")
                        fov : FOV = row['fov_object']
                        timestep = row['timestep']
                        stim = row['stim']
                        channels = row['channels']
                        channels_exposure = row['channels_exposure']
                        stim_profile = row['stim_profile']
                        stim_exposure = row['stim_exposure']

                        metadata_dict = dict(row)
                        metadata_dict['img_type']= ImgType.IMG_RAW
                        metadata_dict['last_channel']= channels[-1]
                                                    
                        # if self._dmd != None:
                        #     metadata_dict['stim_mask'] = self._dmd.sample_mask_on
                            
                        ### Capture the raw image without DMD illumination
                        for i,channel in enumerate(channels):
                            print("kanal")
                            last_channel:bool = i == len(channels)-1
                            metadata_dict['last_channel'] = last_channel
                            metadata_dict['channel'] = channel

                            acquisition_event = useq.MDAEvent(
                                    index= {"t": timestep, "c": i}, # the index of the event in the sequence
                                    channel = {"config":channel, "group":current_channelgroup}, # the channel presets we want to acquire
                                    metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                                    x_pos = fov.pos[0], # only one pos for all channels
                                    y_pos = fov.pos[1],
                                    min_start_time = float(row['time']),
                                    exposure=channels_exposure[i]
                                )
                            print(acquisition_event)

                            #add the event to the acquisition queue
                            self._queue.put(acquisition_event)
                        if stim:
                            while self._queue.qsize() < 1:
                                
                            self._prepare_stim_event(row, current_channelgroup)
                            # print("bin in stim")
                            # threading.Thread(target=self._prepare_stim_event, args=(row, current_channelgroup)).start()
                        #     ### Stimulate using the DMD if stim is True
                        #     stim_mask = fov.stim_mask_queue.get(timeout=10) #wait max 10s for mask
                        #     plt.imshow(stim_mask)
                        #     plt.show()
                        #     #affine transform the mask to the DMD coordinates
                        #     if self._dmd != None:
                        #         stim_mask = self._dmd.affine_transform(stim_mask)

                        #     if np.all(stim_mask == 1): 
                        #         stim_mask = True
                        #     ### expose the image
                        #     metadata_dict['img_type'] = ImgType.IMG_STIM #change the img_type and channels, rest stays the same
                        #     metadata_dict['last_channel'] = True
                        #     metadata_dict['channel'] = stim_profile["channel"]      

                        #     stimulation_event = useq.MDAEvent(
                        #         index= {"t": timestep}, # the index of the event in the sequence
                        #         channel = {"config":stim_profile["channel"], "group":current_channelgroup}, # the channel presets we want to acquire
                        #         metadata = metadata_dict, # (custom) metadata that is attatched to the event/image
                        #         x_pos = row['fov_object'].pos[0], # only one pos for all channels
                        #         y_pos = row['fov_object'].pos[1],
                        #         exposure = stim_exposure, 
                        #         min_start_time=float(row['time']), 
                        #         slm_image=SLMImage(data=stim_mask),
                        #         # properties=[PropertyTuple(stim_profile["device_name"], stim_profile["device_property"], stim_profile["power"])]
                        #     )
                        # self._queue.put(stimulation_event)   
                        

                # # Reached end of acquisition DF
            for fov in df_acquire['fov_object'].unique():
                    fov.tracks = fov.tracks_queue.get(timeout=10) #wait max 10s for tracks
        finally: 
                # # Put the stop event in the queue
            self._queue.put(self.STOP_EVENT)

    def start_mda(self):
        queue_sequence = iter(self._queue.get, self.STOP_EVENT)
        self._mmc.run_mda(queue_sequence)


