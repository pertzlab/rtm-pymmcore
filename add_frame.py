# combines a segmentor, stimulator and tracker into a image processing pipeline. 

from useq import MDAEvent
import numpy as np
import skimage.io
from stimulation import Stim
from segmentation import Segmentator,extract_features
from tracking import Tracker
from utils import MetadataDict,ImgType
import pandas as pd
#import time
import os
from fov import FOV
from utils import labels_to_particles
from utils import write_compressed_pickle
import tifffile

def store_img(img:np.array, metadata:MetadataDict, folder:str):
    """Take the image and store it accordingly. Check the metadata for FOV index and timestamp."""
    fov : FOV = metadata['fov_object']
    img_type = metadata['img_type']
    fname = metadata['fname']
    tifffile.imwrite(os.path.join(fov.path, folder, fname + '.tiff'), img, compression='zlib', compressionargs={'level': 5})

# Create a new pipeline class that contains a segmentator and a stimulator
class ImageProcessingPipeline:
    def __init__(self, segmentator:Segmentator, stimulator:Stim, tracker:Tracker, segmentation_channel:int=0):
        self.segmentator = segmentator 
        self.stimulator = stimulator
        self.tracker = tracker
        self.segmentation_channel = segmentation_channel
    
    def run(self, img: np.ndarray, event: MDAEvent) -> dict:
        """
        Runs the image processing pipeline on the input image.

        Args:
            img (np.ndarray): The input image to process.
            event (MDAEvent): The MDAEvent used to capture the image, which also containins the metadata.

        Returns:
            dict: A dictionary containing the result of the pipeline.

        Pipeline Steps:
        1. Extract metadata from the event object.
        2. Segment the image using the segmentator.
        3. Extract features from the segmented image.
        4. Add frame-related information to the extracted features.
        5. Initialize (frame 0) or run the tracker.
        6. Remove duplicate tracks in the tracker.
        7. If stimulation is enabled, get the stimulated labels and mask.
        8. Store the intermediate tracks dataframe.
        9. Store the segmented images and labels.
        """
        
        # Rest of the code...

        metadata : MetadataDict = event.metadata
        
        fov : FOV = metadata['fov_object']
        df_old = fov.tracks #get the previous table from the FOV- 

        labels = self.segmentator.segment(img[self.segmentation_channel,:,:])
        df_new,labels_rings = extract_features(labels,img)
        df_tracked = self.tracker.track_cells(df_old, df_new, metadata)

        if metadata['stim'] == True:
            stim_mask,labels_stim = self.stimulator.get_stim_mask(labels,metadata)
            fov.stim_mask_queue.put(stim_mask)
            store_img(stim_mask,metadata,'stim_mask')
            #mark in the df which cells have been stimulated
            stim_index = np.where((df_tracked['frame']==metadata['timestep']) & (df_tracked['label'].isin(labels_stim)))[0]
            df_tracked.loc[stim_index,'stim']=True
        else:
            store_img(np.zeros_like(labels).astype(np.uint8),metadata,'stim_mask')
            store_img(np.zeros_like(labels).astype(np.uint8),metadata,'stim')
        
        #store the tracks in the FOV queue
        fov.tracks_queue.put(df_tracked)

        #store the intermediate DF containing the tracks

        #add all the metadata to the DF
        
        if not df_tracked.empty:
            for key, value in metadata.items():
                if type(value) == list:
                # df['new_column'] = df.apply(lambda row: value, axis=1)
                    df_tracked[key] = df_tracked.apply(lambda row: value, axis=1)
                else: 
                    df_tracked.loc[:,key] = value

            #delete unnecessary collumns from the df
            df_tracked = df_tracked.drop('fov_object', axis=1)
            df_tracked = df_tracked.drop('img_type', axis=1)
            df_tracked = df_tracked.drop('channel', axis=1)

        df_tracked.to_parquet(fov.path + "tracks/" + metadata['fname'] + '.parquet')
        # write_compressed_pickle(df_tracked, fov.path + "tracks/" + metadata['fname'])

        particles = labels_to_particles(labels,df_tracked)
        store_img(labels,metadata,'labels')
        store_img(labels_rings,metadata,'labels_rings')
        store_img(particles,metadata,'particles')

        #cleanup: delete the previous pickled tracks file
        # if metadata['timestep'] > 0:
        #     fname_previous = f'{str(fov.index).zfill(3)}_{str(metadata["timestep"]-1).zfill(5)}.pkl'
        #     os.remove(fov.path + "tracks/" + metadata['fname'] + '.zip')


        #TODO return something useful
        #TODO send tracks and stim_mask to the FOV queues


        return {"result": "STOP"}
    
