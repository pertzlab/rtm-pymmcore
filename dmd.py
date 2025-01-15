import skimage
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time
import scipy
#from .acquisition import acq
from pymmcore_plus import CMMCorePlus
from useq import PropertyTuple
from useq._mda_event import SLMImage
from useq import MDAEvent
import random


class DMD():
    '''all methods that relate to the control of the DMD
        img is in camera space (2048px*2048px / 1024px*1024px / ... )
        mask is in dmd space (600px*800px)

    '''
    def __init__(self, mmc: CMMCorePlus, calibration_profile, test_mode :bool= False):
        '''Args:
            mmc: core object from CMMCorePlus()
            test_mode: try the function without a DMD set up in uManager. Defaults to False.
        '''
        #Load all dmd properties from micro-manager
        self.mmc = mmc
        self.test_mode = test_mode
        self.affine = None
        self.calibration_profile = calibration_profile

        if test_mode == False:
            self.name = self.mmc.getSLMDevice()
            self.height = self.mmc.getSLMHeight(self.name)
            self.width = self.mmc.getSLMWidth(self.name)
            self.camera_height = self.mmc.getImageHeight()  
            self.camera_width = self.mmc.getImageWidth()
            self.bppx = self.mmc.getSLMBytesPerPixel(self.name)
            self.exposure_time = self.mmc.getSLMExposure(self.name)
            self.sample_mask_on = np.full((self.height,self.width),255).astype(np.uint8)
            self.sample_mask_off = np.zeros((self.height,self.width)).astype(np.uint8)
        
    def affine_transform(self, img):
        '''Applies transformation matrix on image in camera space. Returns mask in dmd space.
        Args:
            img: image in camera space
            affine: affine transformation matrix
        '''

        if self.affine is None:
            raise ValueError("DMD not calibrated. affine matrix is None. Run calibrate() first.")
        
        img_transformed = skimage.transform.warp(img, self.affine, output_shape=(self.height,self.width), 
                                                 order=None, mode='constant', cval=0.0, clip=True, preserve_range=True)
        img_transformed = img_transformed.astype(np.uint8)
        return img_transformed
    

    def all_on(self):
        '''turn on projector all pixels for a long time
        '''
        self.mmc.setSLMPixelsTo(self.name,255)


    def all_off(self):
        '''turn off pixels
        '''
        self.mmc.setSLMPixelsTo(self.name,0)



    def checker_board(self, pixels = 20):
        '''display a checkerboard pattern for a long time
        '''
        #build checkerboard
        checker_board = (np.indices((self.height, self.width)) // pixels).sum(axis=0) % 2
        checker_board = checker_board.astype(np.uint8) * 255       
        self.display_mask(checker_board)


    def select_well_distributed_points(self, valid_pixels, n_points):
        """
        Select well-distributed points from valid_pixels using a grid-based approach.

        Parameters:
        - valid_pixels (np.ndarray): Array of valid pixel coordinates with shape (N, 2).
        - n_points (int): Number of points to select.
        Returns:
        - selected_points (list of tuples): List of selected (x, y) points.
        """
        selected_points = []

        # Determine grid size based on the number of points
        grid_size = int(np.sqrt(n_points))
        if grid_size ** 2 < n_points:
            grid_size += 1

        # Compute the size of each grid cell
        cell_height = self.height // grid_size
        cell_width = self.width // grid_size

        # Shuffle valid_pixels to ensure random selection within each cell
        shuffled_pixels = valid_pixels.copy()
        np.random.shuffle(shuffled_pixels)

        for i in range(grid_size):
            for j in range(grid_size):
                if len(selected_points) >= n_points:
                    break

                # Define the boundaries of the current cell
                row_start = i * cell_height
                row_end = (i + 1) * cell_height if i < grid_size - 1 else self.height
                col_start = j * cell_width
                col_end = (j + 1) * cell_width if j < grid_size - 1 else self.width

                # Find valid pixels within the current cell
                cell_pixels = shuffled_pixels[
                    (shuffled_pixels[:, 0] >= row_start) & (shuffled_pixels[:, 0] < row_end) &
                    (shuffled_pixels[:, 1] >= col_start) & (shuffled_pixels[:, 1] < col_end)
                ]

                if len(cell_pixels) > 0:
                    # Select a random pixel from the cell
                    selected_point = tuple(cell_pixels[random.randint(0, len(cell_pixels) - 1)])
                    selected_points.append(selected_point)

        # If not enough points are selected, randomly select remaining points from all valid_pixels
        if len(selected_points) < n_points:
            remaining = n_points - len(selected_points)
            additional_points = random.sample(list(map(tuple, valid_pixels)), remaining)
            selected_points.extend(additional_points)

        return selected_points

    def calibrate(self, verbous=False, n_points = 15, radius = 4, exposure = 800, marker_style = 'x', calibration_points_DMD = None):
        
        '''Calibrate the dmd and camera coordinate systems. 
        Projects 3 points in DMD space and detects them in camera space, 
        then finds the affine transofmation matrix. 
        Args:
            verbous (bool, optional): Whether to display additional images during calibration. Defaults to False.
            blur (int, optional): Blur size for captured images. Defaults to 10.
            circle_size (int, optional): Size of the calibration circle projected. Defaults to 10.
            marker_style (str, optional): Marker style for calibration points. Defaults to 'x'.
            calibration_points_DMD (list, optional): List of X/Y DMD calibration points. Defaults to [(180,180),(700,130),(180,550)], which works well on our (800x600 DMD).
        '''
        # good working points:  ([250, 380], [100,800], [900, 800], [250, 800], [490,380],[500,400],[1000,340])
        src = []
        dst = []
        event_p = []
        events = []
        calibration_images = []

        img_dmd_full = (np.ones((self.height,self.width))*255).astype(np.uint8)
        img_dmd_full_w_borders = img_dmd_full.copy()
        img_dmd_full_w_borders[0:100] = 0
        img_dmd_full_w_borders[:, 0:340] = 0
        img_dmd_full_w_borders[-100:] = 0
        img_dmd_full_w_borders[:, -50:] = 0

        valid_pixels = np.array(np.where(img_dmd_full_w_borders>0)).T

        if calibration_points_DMD is None:
            calibration_points_DMD = []
            calibration_points_DMD = self.select_well_distributed_points(valid_pixels, n_points)
        
        for p in calibration_points_DMD: 
            img_p = np.zeros((self.height, self.width)).astype(np.uint8)
            src.append((p[1],p[0]))
            rr, cc = skimage.draw.disk((p[0],p[1]), radius)
            img_p[rr, cc] = 255
            event_p = MDAEvent(slm_image=SLMImage(data=img_p, device=self.name), exposure=exposure, 
                               channel={"config":self.calibration_profile["channel_config"], "group":self.calibration_profile["channel_group"]}, 
                               properties=[PropertyTuple(self.calibration_profile["device_name"], self.calibration_profile["property_name"], self.calibration_profile["power"])])
            events.append(event_p)

        self.mmc.mda.events.frameReady.disconnect()
        @self.mmc.mda.events.frameReady.connect
        def new_frame(img: np.ndarray, event: MDAEvent):
            calibration_images.append(img)

        self.mmc.mda.run(events)
        calibration_images = np.array(calibration_images)
   
        for img in calibration_images:
            img = skimage.filters.gaussian(img, sigma=1)
            max_x = np.argmax(img.max(axis=0))
            max_y = np.argmax(img.max(axis=1))
            dst.append((max_x,max_y))
        
        src = np.array(src)
        dst = np.array(dst)

        affine_model, inliers = skimage.measure.ransac((src, dst), skimage.transform.AffineTransform, min_samples=3, 
                                                residual_threshold=2, max_trials=5000)
    
        if np.sum(inliers) < 5:
            self.mmc.mda.events.frameReady.disconnect()
            self.mmc.mda.run([MDAEvent(slm_image=SLMImage(data=True,device=self.name),exposure=1, 
                            properties=[PropertyTuple(self.calibration_profile["device_name"], self.calibration_profile["property_name"], 0)])])


            raise ValueError("Not enough inliers found for calibration. Try again with a different FOV.")
        self.affine = affine_model.params

        if verbous:
            # test the calibration on three new points
            event_p = []
            events = []
            test_image = []
            test_src = []
            test_dst = []
            p0, p1, p2 = ([350, 500], [400,800], [700, 800])
            for p in [p0,p1,p2]:
                img_p = np.zeros((self.camera_height,self.camera_width)).astype(np.uint8)
                rr, cc = skimage.draw.disk((p[0],p[1]), 30)
                test_src.append((p[1],p[0]))
                img_p[rr,cc] = 255
                img_warp = self.affine_transform(img_p)
                event_p = MDAEvent(slm_image=SLMImage(data=img_warp,device=self.name),exposure=exposure, 
                               channel={"config":self.calibration_profile["channel_config"], "group":self.calibration_profile["channel_group"]}, 
                               properties=[PropertyTuple(self.calibration_profile["device_name"], self.calibration_profile["property_name"], self.calibration_profile["power"])])
                events.append(event_p)

            self.mmc.mda.events.frameReady.disconnect()
            @self.mmc.mda.events.frameReady.connect
            def new_frame(img: np.ndarray, event: MDAEvent):
                test_image.append(img)

            self.mmc.mda.run(events)
            calibration_images = np.array(calibration_images)
            for img in test_image:
                img = skimage.filters.gaussian(img, sigma=1)
                max_x = np.argmax(img.max(axis=0))
                max_y = np.argmax(img.max(axis=1))
                test_dst.append((max_x,max_y))
            
            test_src = np.array(test_src)
            test_dst = np.array(test_dst)

            fig, axs = plt.subplots(figsize=(20, 4), ncols=4, dpi = 250)
            axs[0].imshow(calibration_images[0], cmap='gray')
            axs[0].scatter(dst[0][0],dst[0][1], marker= "x", facecolors='red')
            axs[1].imshow(calibration_images[1], cmap='gray')
            axs[1].scatter(dst[1][0],dst[1][1],marker = "x", facecolors='red')
            
            axs[2].imshow(test_image[0], cmap='gray')
            axs[2].scatter(test_src[0][0],test_src[0][1],marker = "x", facecolor='green')
            for i in range(3):
                axs[3].scatter(test_src[i][0],test_src[i][1],marker = "x", facecolor='red')
                axs[3].scatter(test_dst[i][0],test_dst[i][1],marker = "x", facecolor='green')


            plt.show()
        self.mmc.mda.events.frameReady.disconnect()
        self.mmc.mda.run([MDAEvent(slm_image=SLMImage(data=True,device=self.name),exposure=1, 
                        properties=[PropertyTuple(self.calibration_profile["device_name"], self.calibration_profile["property_name"], 0)])])





    # def display_mask(self,mask):
    #     '''Display the mask loaded on the dmd. Displays it for the set exposure.
    #     '''
 
    #     self.mmc.setSLMImage(self.name, mask)

    #     self.mmc.displaySLMImage(self.name)

    # def set_exposure(self, exposure_time):
    #     '''Set the time a mask is displayed on the dmd.
    #     '''
    #     #todo: check for max allowed exposure time
    #     self.exposure_time = exposure_time
    #     self.mmc.setSLMExposure(self.name,exposure_time)

    # def capture_and_stim_mask(self, mask, dmd_exposure_time,camera_exposure_time,delay=0):
    #     '''Stimulate using dmd and take an image.
    #     '''    
    #     #stimulate using dmd and take an image of the stim
    #     self.mmc.setSLMExposure(self.name, dmd_exposure_time)#set exposure dmd
    #     self.mmc.setExposure(camera_exposure_time)#set exposure camera
    #     self.upload_mask(mask) #upload mask to dmd        
    #     self.display_mask(mask)#display on dmd
    #     time.sleep(delay)#time between stimulation begin and capture begin
    #     self.mmc.snapImage()#take picture
        
    #     #TODO check if this works, then delete tagged_img references
    #     img = self.mmc.getImage()

    #     #tagged_img = self.mmc.getTaggedImage()
    #     #img_height = tagged_img.tags['Height']
    #     #img_width = tagged_img.tags['Width']
    #     #img = tagged_img.pix.reshape(img_height, img_width)

    #     return img

    # def capture_and_stim_full_on(self, dmd_exposure_time,camera_exposure_time,delay=0):
    #     '''Stimulate using dmd (all pixels on) and take an image.
    #     '''    
    #     #stimulate using dmd and take an image of the stim
    #     self.mmc.setSLMExposure(self.name, dmd_exposure_time)#set exposure dmd
    #     self.mmc.setExposure(camera_exposure_time)#set exposure camera
    #     self.mmc.setSLMPixelsTo(self.name, 1)
    #     self.mmc.displaySLMImage(self.name)
    #     time.sleep(delay)#time between stimulation begin and capture begin
    #     self.mmc.snapImage()#take picture
    #     tagged_img = self.mmc.getTaggedImage() #TODO check if this is now fixed
    #     img_height = tagged_img.tags['Height']
    #     img_width = tagged_img.tags['Width']
    #     img = tagged_img.pix.reshape(img_height, img_width)        
    #     return img


    # def capture_and_stim_img(self, img, affine, dmd_exposure_time,camera_exposure_time,delay=0):
    #     '''Transform img using affine matrix, then stimulate using dmd and take an image.
    #     ''' 
    #     #mask = scipy.ndimage.affine_transform(img, affine, output_shape=(self.width, self.height))
    #     mask = cv2.warpAffine(img, affine, (self.width, self.height))
    #     img_captured = self.capture_and_stim_mask(mask, dmd_exposure_time,camera_exposure_time, delay)
    #     return img_captured


    # def transform_and_disp(self, img, affine):
    #     '''Transform img using affine matrix, then uplaod and display it.
    #     ''' 
    #     mask = cv2.warpAffine(img, affine, (self.width, self.height))
    #     #self.core.set_slm_exposure(self.name, 20000)#set exposure dmd    
    #     self.display_mask(mask)#display on dmd


