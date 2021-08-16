import os 

import cv2 
import numpy as np 

from .eye import Eye
from .calibration import Calibration



class GazeTracking(object):
    """ This class tracks the user's gaze.
        It provides useful information like: 
            - the position of the eyes and pupils
    """

    def __init__(self):
        self.frame = None
        self.face_landmarks = None

        self.eye_left = None
        self.eye_right = None

        self.calibration = Calibration()



    def refresh(self, frame:np.ndarray, face_landmarks:list):
        """ Refreshes the frame and analyzes it.
        
            Arguments:
                frame : The frame to analyze
                face_landmarks : 468-landmark coordinates for face 
        """
        self.frame = frame
        self.face_landmarks = face_landmarks
        self._analyze()



    def _analyze(self):
        """ Detects the face and initialize Eye objects
        """
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        

        try:
            self.eye_left = Eye(frame, self.face_landmarks , 0, self.calibration)
            self.eye_right = Eye(frame, self.face_landmarks , 1, self.calibration)


        except IndexError:
            self.eye_left = None
            self.eye_right = None       



    def pupil_left_coords(self):
        """ Returns the coordinates of the left pupil
        """
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    @property
    def pupils_located(self):
        # @property ; (ref) https://dojang.io/mod/page/view.php?id=2476
        #           ; (ref) https://www.daleseo.com/python-property/

        """ Check that the pupils have been located
        """
        
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        
        except Exception:
            return False            
