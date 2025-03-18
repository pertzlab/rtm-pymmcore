import numpy as np
from segmentation import Segmentator
import skimage
from napari_convpaint import conv_paint, conv_paint_utils


from imaging_server_kit import Client

client = Client("http://localhost:8000")

print(client.algorithms)
# [`rembg`, `stardist`, `cellpose`]

algo_output = client.run_algorithm(
    algorithm="rembg",
    image=(...),
    rembg_model_name="silueta",
)


import matplotlib.pyplot as plt

"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SegmentatorConvpaint(Segmentator):

    def __init__(self, model_path: str = "2D_versatile_fluo", min_size=0):

        self.random_forest, self.model, self.model_param, self.model_state = (
            conv_paint.load_model(model_path)
        )
        self.min_size = min_size

    def segment(self, img: np.ndarray) -> np.ndarray:
        """
        Run the stardist model on data and do post-processing (remove small cells)
        """

        mean, std = conv_paint_utils.compute_image_stats(img)
        img_normed = conv_paint_utils.normalize_image(img, mean, std)
        labels = self.model.predict_image(
            img_normed, self.random_forest, self.model_param
        )
        if self.min_size > 0:
            # remove cells below threshold
            labels = skimage.morphology.remove_small_objects(
                labels, min_size=self.min_size, connectivity=1
            )
        labels = skimage.measure.label(labels, background=1)
        return labels
