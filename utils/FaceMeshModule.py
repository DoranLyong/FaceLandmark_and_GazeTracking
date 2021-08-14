""" (ref) https://www.computervision.zone/lessons/code-files-16/
    (ref) 
"""
import time 

import cv2 
import numpy as np
import mediapipe as mp

from .imtools import ( gaussian_kernel,

                        )




class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(   self.staticMode, self.maxFaces,
                                                    self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []

        blank_img = np.zeros_like(img)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                                self.drawSpec, self.drawSpec) # (ref) https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
                    self.mpDraw.draw_landmarks(blank_img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                                self.drawSpec, self.drawSpec)
                face = []

                ih, iw, ic = img.shape
                cy_min, cx_min = ih, iw
                cy_max, cx_max = 0, 0 



                for id, lm in enumerate(faceLms.landmark):
#                    print(lm)
                    print(id)
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
#                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
#                               0.7, (0, 255, 0), 1)

#                    print(id,x,y)  # (Landmark_num , x_coordi, y_coordi)
                    face.append([x,y,z])



                    # ========================= # 
                    # Get the face bounding box # 
                    # ========================= #
                    #  (ref) https://github.com/google/mediapipe/issues/1737  

                    if x < cx_min:
                        cx_min = x 
                    
                    if y < cy_min:
                        cy_min = y
                    
                    if x > cx_max:
                        cx_max = x 
                    if y > cy_max:
                        cy_max = y 

                cv2.rectangle(img, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
                cv2.rectangle(blank_img, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)


                heatmap = draw_heatmap(face, *[cx_min, cy_min, cx_max, cy_max])


                faces.append(face)
        return img, blank_img,faces





def draw_heatmap(face_landmarks:list, *bbox):

    cx_min, cy_min, cx_max, cy_max = bbox

    h = cy_max  - cy_min + 1 
    w = cx_max  - cx_min + 1 
    
    portrait = np.zeros( [h, w, 3], np.uint8 )


    raw_landmark = np.array(face_landmarks) # (x, y, z )
    translation = np.array([cx_min, cy_min, 0])

    landmark = raw_landmark - translation   # zero-centroid


    for x, y ,z in landmark:
        cv2.circle(portrait, (x,y), 1, (0,255,0), -1)


    cv2.imshow("portrait", portrait)
    



def iris_detector():
    """ (ref) https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
    """




