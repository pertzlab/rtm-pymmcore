import numpy as np
import numpy.typing as npt
from skimage.draw import disk
from skimage.measure import regionprops
from fov import FOV


#TODO return also labels_stim, list in which the stimulated cells are marked
class Stim:
    """
    Base class for all stimulators. Specific implementations should inherit 
    from this class and override the get_stim_mask method.
    """
    def get_stim_mask(self, label_image: np.ndarray) -> npt.NDArray[np.uint8]:
        """
        Parameters:
        label_image (np.ndarray): The label image to stimulate.

        Returns:
        np.ndarray: The stimulation mask.
        list: A list of labels that were stimulated.
        """
        raise NotImplementedError("Subclasses should implement this!")

class StimWholeFOV(Stim):
    """
    Stimulate the whole FOV.
    """
    def get_stim_mask(self, label_image: np.ndarray, metadata:dict) -> npt.NDArray[np.uint8]:
        return np.ones_like(label_image), [1,2,3,4] #some dummy values

class StimNothing(Stim):
    """Use when you don't want to stimulate. Returns empty stimulation mask."""
    def get_stim_mask(self, label_image: np.ndarray) -> npt.NDArray[np.uint8]:
        return np.zeros_like(label_image), [1,2,3,4] #some dummy values

class StimCircle(Stim):
    """
    Circle stimulator.

    This class implements a simple circle stimulation. It creates a stimulation mask
    by drawing a circle at the centroid of each labeled region in the label image.
    The radius of the circle and the x/y offset of the centroid can be parametrized.
    """
    def get_stim_mask(self, label_image: np.ndarray, metadata:dict) -> npt.NDArray[np.uint8]:

        fov = metadata['fov_object']
        offset_x = metadata['offset_x']
        offset_y = metadata['offset_y']
        radius = metadata['radius']

        stim_mask = np.zeros_like(label_image, dtype=np.uint8)
        props = regionprops(label_image)
        labels_stim = []
        for prop in props:
            centroid = (prop.centroid[0] + offset_y, prop.centroid[1] + offset_x)
            rr, cc = disk(centroid, radius=radius)
            stim_mask[rr, cc] = 1
            labels_stim.append(prop.label)
        return stim_mask, labels_stim
    
class StimExtraParameters(Stim):
    """
    An example class that uses the metadata from a FOV to do more advanced stimulation
    patterns.
    """
    def get_stim_mask(self, label_image: np.ndarray,  metadata:dict) -> npt.NDArray[np.uint8]:

        if 'stim_property' in metadata['fov_object'].properties:
            # Do something if 'stim_property' is in the dict
            return np.ones_like(label_image).astype(np.uint8), []

        else:
            # Property is not in dict!
            raise KeyError("'stim_property' is missing in fov.properties")
    

#Helper functions
def coordinates_to_lightmap(xy, mask):
    '''Takes a list of coordinates [(y,x),(y,x),...] and draws an ellipse on a mask for every point. '''
    import cv2

    #Elipse properties
    axesLength = (3, 3) 
    angle = 0
    startAngle = 0
    endAngle = 360
    color = (1) 
    thickness = -1
    light_mask = np.zeros_like(mask)        
    center_coordinates = (int(xy[1]),int(xy[0]))
    light_mask = cv2.ellipse(light_mask, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 

    return light_mask