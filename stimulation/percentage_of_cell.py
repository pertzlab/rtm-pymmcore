from .base_stim import Stim
import numpy as np
import skimage
import math


class StimPercentageOfCell(Stim):
    """
    Stimulate a percentage of the cell.

    This class implements a stimulation that stimulates a percentage of the cell.
    The percentage can be parametrized.
    """

    def above_line(self, i, j, x2, y2, x3, y3):
        v1 = (x2 - x3, y2 - y3)
        v2 = (x2 - i, y2 - j)
        xp = v1[0] * v2[1] - v1[1] * v2[0]
        return xp > 0

    def get_stim_mask(
        self, label_image: np.ndarray, metadata: dict = None, img: np.array = None
    ) -> np.ndarray:
        light_map = np.zeros_like(label_image)
        props = skimage.measure.regionprops(label_image)
        percentage_of_stim = metadata.get("treatment", {}).get(
            "stim_cell_percentage", 0.3
        )

        try:
            extent = 0.5 - percentage_of_stim
            for prop in props:
                label = prop.label
                single_label = label_image == label

                orientation = prop.orientation
                y0, x0 = prop.centroid
                orientation = prop.orientation

                # Find point where cutoff line and major axis intersect
                x2 = x0 - math.sin(orientation) * extent * prop.major_axis_length
                y2 = y0 - math.cos(orientation) * extent * prop.major_axis_length

                # find second point on line
                length = 0.5 * prop.minor_axis_length
                x3 = x2 + (length * math.cos(-orientation))
                y3 = y2 + (length * math.sin(-orientation))

                # make mask where all pixels above line are TRUE
                cutoff_mask = np.fromfunction(
                    lambda i, j: self.above_line(j, i, x3, y3, x2, y2),
                    np.shape(label_image),
                    dtype=int,
                )

                frame_labeled_expanded = skimage.segmentation.expand_labels(
                    single_label, 5
                )
                stim_mask = np.logical_and(cutoff_mask, frame_labeled_expanded)

                light_map = np.logical_or(light_map, stim_mask)
                light_map = light_map.astype("uint8")
            return light_map, [1, 2, 3, 4]  # some dummy values
        except Exception as e:
            print(e)
            return np.zeros_like(label_image), [1, 2, 3, 4]
