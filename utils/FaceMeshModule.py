""" (ref) https://www.computervision.zone/lessons/code-files-16/
    (ref) 
"""
import time 

import cv2 
import numpy as np
import mediapipe as mp
import albumentations as A
    

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
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # ============== #  
        # Albumentations # 
        # ============== #  
        self.transform = A.Compose( [ A.Resize(height=224, width=224),  # (ref) https://www.programcreek.com/python/example/120573/albumentations.Resize
                                                                    ],    # (ref)https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
                                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, angle_in_degrees=True) # (ref) https://albumentations.ai/docs/getting_started/keypoints_augmentation/ 
                                    )


    def findFaceMesh(self, img, draw=True):
        raw_img = img.copy()
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces_kps = []
        onlyface_kps = []
        onlyfaces = [] 

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
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
#                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
#                               0.7, (0, 255, 0), 1)

#                    print(id,x,y)  # (Landmark_num , x_coordi, y_coordi)
                    face.append([x,y])



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


                # ================================== # 
                #     Get the face area in resize    # 
                # ================================== #
                if faceLms.landmark:
                    
                    cropped_face = raw_img[cy_min:cy_max+1, cx_min:cx_max+1]
                    cropped_face = cv2.resize(cropped_face, dsize=(224,224))  
                    

                    transformed_kps = self.kp_transform(face, *[cx_min, cy_min, cx_max, cy_max] )

                    onlyfaces.append(cropped_face)
                    onlyface_kps.append(transformed_kps)

                faces_kps.append(face)

        return img, blank_img,faces_kps, onlyface_kps, onlyfaces





    def kp_transform(self, face_landmarks:list, *bbox ):

        # ==================================== # 
        # Keypoints(=landmarks) transformation # 
        # ==================================== # 
        # (ref) https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        # (ref) https://imgaug.readthedocs.io/en/latest/source/jupyter_notebooks.html

        cx_min, cy_min, cx_max, cy_max = bbox

        h = cy_max  - cy_min + 1 
        w = cx_max  - cx_min + 1 

        portrait = np.zeros( [h, w], np.uint8 )


        raw_landmark = np.array(face_landmarks) # (x, y )
        translation = np.array([cx_min, cy_min])

        landmark = raw_landmark - translation   # zero-centroid

        transformed = self.transform(   image = portrait, 
                                        keypoints= landmark.tolist(), # (ref) https://appia.tistory.com/175
                                    )


        return transformed['keypoints']




