""" (ref) https://github.com/Rassibassi/mediapipeDemos/blob/main/iris.py
"""

import cv2
import numpy as np
import mediapipe as mp

from .iris_lm_depth import from_landmarks_to_depth



class IrisTracking(object):
    """ This class tracks the user's iris.
        It provides useful information like: 
            - the position of the eyes and pupils
    """    

    points_idx = sorted((set([33, 133, 362, 263, 61, 291, 199])))

    left_eye_landmarks_id = np.array([33, 133])
    right_eye_landmarks_id = np.array([362, 263])

    dist_coeff = np.zeros((4, 1))




    def __init__(self):
        self.frame = None
        self.face_landmarks = None

        self.eye_left = None
        self.eye_right = None

        # pseudo camera internals
        self.focal_length = None   # set it as frame width 

        self.smooth_left_depth = -1
        self.smooth_right_depth = -1
        self.smooth_factor = 0.1        


    def refresh(self, frame:np.ndarray, face_landmarks:np.ndarray):
        """ Refreshes the frame and analyzes it.
        
            Arguments:
                frame : The frame to analyze
                face_landmarks : 468-landmark coordinates for face 
        """
        h, w = frame.shape[:2]
        self.frame = frame
        self.face_landmarks = face_landmarks.T

        self.focal_length = w 

        self._analyze()       



    def _analyze(self):
        """ Detects the face and initialize Eye objects
        
            from_landmarks_to_depth() returns like:     

            (   left_depth,
                left_iris_size,
                left_iris_landmarks,
                left_eye_contours,
            ) 

            (   right_depth,
                right_iris_size,
                right_iris_landmarks,
                right_eye_contours,
            )
        """         
        
        
        try:
            h, w = self.frame.shape[:2] 

            self.eye_left = from_landmarks_to_depth(    self.frame,  # RGB order frame 
                                                        self.face_landmarks[:, self.left_eye_landmarks_id],
                                                        (w, h),     # image size 
                                                        is_right_eye=False,
                                                        focal_length=self.focal_length,
                                                    )

            self.eye_right = from_landmarks_to_depth(   self.frame,  # RGB order frame 
                                                        self.face_landmarks[:, self.right_eye_landmarks_id],
                                                        (w, h),     # image size 
                                                        is_right_eye=True,
                                                        focal_length=self.focal_length,
                                                    )

        except IndexError:
            self.eye_left = None, None, None, None 
            self.eye_right = None, None, None, None 


    def left_iris(self) -> np.ndarray: 

        (   depth, 
            iris_size,
            iris_landmarks,
            eye_contours, 
        ) =  self.eye_left 
        
        if self.smooth_left_depth < 0 : 
            self.smooth_left_depth = depth

        elif self.smooth_left_depth >= 0:
            self.smooth_left_depth =  ( self.smooth_left_depth * (1 - self.smooth_factor) + depth * self.smooth_factor ) 

#        print(f"left_depth in cm: {self.smooth_left_depth / 10:.2f}")
#        print(f"left_iris_size: {iris_size:.2f}")

#        print(iris_landmarks.shape) # (5, 3) shape with (x, y, z) order 
        return iris_landmarks
        

    def right_iris(self) -> np.ndarray:
        (   depth, 
            iris_size,
            iris_landmarks,
            eye_contours, 
        ) =  self.eye_right 
        
        if self.smooth_right_depth < 0 : 
            self.smooth_right_depth = depth

        elif self.smooth_right_depth >= 0:
            self.smooth_right_depth =  ( self.smooth_right_depth * (1 - self.smooth_factor) + depth * self.smooth_factor ) 
        
#        print(f"right_depth in cm: {self.smooth_right_depth / 10:.2f}")
#        print(f"right_iris_size: {iris_size:.2f}")

#        print(iris_landmarks.shape) # (5, 3) shape with (x, y, z) order 
        return iris_landmarks