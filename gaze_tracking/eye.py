""" (ref) https://github.com/antoinelame/GazeTracking/blob/79b160cbb5f2fb1840f17781316d7aceb4b02651/gaze_tracking/eye.py#L7
"""
import math
from operator import itemgetter  

import numpy as np 
import cv2 

from .pupil import Pupil




class Eye(object):
    """ This class creates a new frame to isolate the eye and initiates the pupil detection.
    """
    # ============== # 
    # Get eye points # 
    # ============== # 
    # 68-landmark example; (ref) https://stackoverflow.com/questions/67362053/is-there-a-way-to-select-a-specific-point-at-the-face-after-detecting-facial-lan
    # 468-landmark example; (ref) https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py
    #                       (ref) https://stackoverflow.com/questions/66649492/how-to-get-specific-landmark-of-face-like-lips-or-eyes-using-tensorflow-js-face
    # face mesh image ; (ref) https://github.com/google/mediapipe/issues/1615

    LEFT_EYE_POINTS = sorted([263, 249, 390, 373, 374, 380, 381, 382, 466, 388, 387, 386, 385, 384, 398, 362])  # zero-indexing 
    RIGHT_EYE_POINTS = sorted([33, 7, 163, 144, 145, 153, 154, 155, 246, 161, 160, 159, 158, 157, 173, 133])




    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None   # set as np.ndarray   


        self._analyze(original_frame, landmarks, side, calibration)





    def _analyze(self, original_frame:np.ndarray, landmarks: list, side, calibration):
        """ Detects and isolates the eye in a new frame, sends data to the calibration and initializes Pupil object.

            Arguments: 
                original_frame : Frame passed by the user
                landmarks : Facial landmarks for the face region from the face mesh of google mediapipe 
                side: Indicates whether it's the left eye (0) or the right eye (1)
                calibration (calibration.Calibration): Manages the binarization threshold value
        """

        if side == 0:
            points = self.LEFT_EYE_POINTS        

        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return


        self._isolate(original_frame, landmarks, points)           

        if not calibration.is_complete(): 
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)





    def _isolate(self, frame:np.ndarray, landmarks:list, points:list):
        """ Isolate an eye, to have a frame without other part of the face.

            Arguments:
                frame : Frame containing the face
                landmarks : Facial landmarks for the face region
                points : Point indices of an eye (from the 468 landmarks)
        """
        region = np.array(itemgetter(*points)(landmarks), dtype=np.int32)   # (x, y)-order ; (ref) https://stackoverflow.com/questions/18272160/access-multiple-elements-of-list-knowing-their-index
                                                                            # (N, 2)  shape
        self.landmark_points = region 


        # =================================== # 
        # Applying a mask to get only the eye # 
        # =================================== # 
        height, width = frame.shape[:2] 
        black_frame = np.zeros((height, width), np.uint8)
        
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)


        # =================================== # 
        #           Cropping on the eye       #
        # =================================== # 
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)
