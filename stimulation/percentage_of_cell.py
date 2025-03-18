from .base_stim import Stim
import numpy as np


class StimPercentageOfCell(Stim):
    """
    Stimulate a percentage of the cell.

    This class implements a stimulation that stimulates a percentage of the cell.
    The percentage can be parametrized.
    """

    def get_stim_mask(self, label_image: np.ndarray, metadata: dict) -> np.ndarray:
        fov = metadata["fov_object"]
        percentage = metadata["percentage"]

        stim_mask = np.zeros_like(label_image, dtype=np.uint8)
        props = regionprops(label_image)
        labels_stim = []

        for prop in props:
            if prop.area > 0:
                stim_area = int(prop.area * percentage / 100)
                stim_mask[stim_area] = 1
                labels_stim.append(prop.label)

        return stim_mask, labels_stim


# # stim percentages
# def above_line(i, j, x2, y2, x3, y3):
#     """Check if point (i,j) is above the line given py points (x2,y2) and (x3,y3)"""
#     v1 = (x2 - x3, y2 - y3)  # vector along line
#     v2 = (x2 - i, y2 - j)  # vector petween line point and point to check
#     xp = v1[0] * v2[1] - v1[1] * v2[0]  # cross product
#     return xp > 0


# def spot_mask_from_labels(labels,percentage):
#     '''Stim certain percentage of cell along major axis.
#     50%: stim one half of the cell with border lying on minor axis.'''

#     light_map = np.zeros_like(labels)
#     props = skimage.measure.regionprops(labels)

#     extent = 0.5-percentage

#     for prop in props:
#         label =prop.label
#         single_label = (labels == label)
#         bbox = prop.bbox
#         orientation = prop.orientation

#         y0, x0 = prop.centroid
#         orientation = prop.orientation

#         #Find point where cutoff line and major axis intersect
#         x2 = x0 - math.sin(orientation) * extent * prop.major_axis_length
#         y2 = y0 - math.cos(orientation) * extent * prop.major_axis_length


#         #find second point on line
#         length = 0.5 * prop.minor_axis_length
#         x3 = x2 + (length * math.cos(-orientation))
#         y3 = y2 + (length * math.sin(-orientation))

#         #make mask where all pixels above line are TRUE
#         cutoff_mask =  np.fromfunction(lambda i, j: above_line(j,i,x3,y3,x2,y2), np.shape(labels), dtype=int)
#        # plt.imshow(single_label)
#        # plt.scatter(x0,y0)

#         #plt.scatter(x2,y2)
#         #plt.scatter(x3,y3)
#         #plt.show()
#         #intersect with cell mask

#         frame_labeled_expanded = expand_labels(single_label,5)
#         stim_mask = np.logical_and(cutoff_mask, frame_labeled_expanded)

#         light_map = np.logical_or(light_map, stim_mask)
#         light_map = light_map.astype('uint8')

#     return light_map

# plt.imshow(spot_mask_from_labels(masks,0.3))
