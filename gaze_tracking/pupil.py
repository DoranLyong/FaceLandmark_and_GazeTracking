""" (ref) https://github.com/antoinelame/GazeTracking/blob/79b160cbb5f2fb1840f17781316d7aceb4b02651/gaze_tracking/pupil.py#L5
"""
import numpy as np 
import cv2 




#%% 
class Pupil(object):
    """ This class detects the iris of an eye and estimates the position of the pupil 
    """

    def __init__(self, eye_frame:np.ndarray, threshold:int):
        self.iris_frame = None 
        self.threshold = threshold 

        self.x = None 
        self.y = None 

        self.detect_iris(eye_frame)


    def detect_iris(self, eye_frame:np.ndarray):
        """ Detects the iris and estimates the position of the iris by calculating the centroid.

            Arguments: 
                eye_frame: Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _= cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]    # (ref) https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
                                                                                                    # (ref) https://opencv-python.readthedocs.io/en/latest/doc/16.imageContourFeature/imageContourFeature.html#goal

        contours = sorted(contours, key=cv2.contourArea)    # key; (ref) https://blockdmask.tistory.com/466
                                                            # cv2.contourArea; (ref) https://opencv-python.readthedocs.io/en/latest/doc/16.imageContourFeature/imageContourFeature.html
                                                            #                  (ref) https://docs.opencv.org/4.5.2/dd/d49/tutorial_py_contour_features.html
                                                            #                  (ref) https://deep-learning-study.tistory.com/232

        try: 
            moments = cv2.moments(contours[-2]) # to get contours' center; (ref) https://076923.github.io/posts/Python-opencv-25/
                                                #                          (ref) https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])                                                

        except (IndexError, ZeroDivisionError):
            pass 
        


    @staticmethod
    def image_processing(eye_frame:np.ndarray, threshold:int) -> np.ndarray:
        # static method; (ref) https://wikidocs.net/21054
        #                (ref) https://hckcksrl.medium.com/python-%EC%A0%95%EC%A0%81%EB%A9%94%EC%86%8C%EB%93%9C-staticmethod-%EC%99%80-classmethod-6721b0977372
        """ Performs operations on the eye frame to isolate the iris

            Arguments:
                eye_frame : Frame containing an eye and nothing else
                threshold : Threshold value used to binarize the eye frame
        """

        kernel = np.ones((3, 3), np.uint8)

        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)  # (ref) https://jvvp.tistory.com/1032
                                                                # (ref) https://opencv-python.readthedocs.io/en/latest/doc/11.imageSmoothing/imageSmoothing.html

        new_frame = cv2.erode(new_frame, kernel, iterations=3)  # (ref) https://opencv-python.readthedocs.io/en/latest/doc/12.imageMorphological/imageMorphological.html?highlight=erosion
                                                                # (ref) https://deep-learning-study.tistory.com/226

        ret, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY) # (ref) https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html     

        return new_frame                                                           


