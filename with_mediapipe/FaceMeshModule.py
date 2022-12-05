""" (ref) https://www.computervision.zone/lessons/code-files-16/
    (ref) 
"""
import time 

import cv2 
import numpy as np
import mediapipe as mp
import albumentations as A

from .iris import IrisTracking
    




class FaceMeshDetector():

    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    SMALL_CIRCLE_SIZE = 1
    LARGE_CIRCLE_SIZE = 2    

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        
        self.faceMesh = self.mpFaceMesh.FaceMesh(   static_image_mode=self.staticMode, 
                                                    max_num_faces=self.maxFaces,
                                                    refine_landmarks=False,
                                                    min_detection_confidence=self.minDetectionCon, 
                                                    min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        self.iris  = IrisTracking()

        # ============== #  
        # Albumentations # 
        # ============== #  
        self.transform = A.Compose( [ A.Resize(height=256, width=256),  # (ref) https://www.programcreek.com/python/example/120573/albumentations.Resize
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


                face = []

                ih, iw, ic = img.shape
                cy_min, cx_min = ih, iw
                cy_max, cx_max = 0, 0                 

                # ========================= # 
                #         Get iris          # 
                # ========================= # 
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in faceLms.landmark])

                self.iris.refresh(self.imgRGB, landmarks)
                left_irisLM = self.iris.left_iris()  # (5, 3) shape with (x, y, z) order 
                right_irisLM = self.iris.right_iris() # those are normalized coordinates; (0, 1)

                iris_landmarks = np.concatenate([   right_irisLM, 
                                                    left_irisLM,   
                                                ]).tolist()  # np.ndarray -> list 

                iris_landmarks = [ (np.array((iw, ih , 1)) * irisLM).astype(np.float32) for irisLM in  iris_landmarks ] # Denormalize the iris position 


                for x, y, z in iris_landmarks:
                    cv2.circle(img, (int(x),int(y)), self.LARGE_CIRCLE_SIZE, self.YELLOW, -1)


                # ========================= # 
                #      Draw Face Mesh       #  
                # ========================= # 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, # (ref) https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py
                                                self.drawSpec, self.drawSpec) # (ref) https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
                    
                    self.mpDraw.draw_landmarks(blank_img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                self.drawSpec, self.drawSpec)


                faceMesh_landmarks = [ ( np.array((iw, ih , 1)) * np.array((lm.x, lm.y, lm.z ))).astype(np.float32) for lm in faceLms.landmark ] # Denormalize the face mesh position 

                faceMesh_landmarks.extend(iris_landmarks) # iris + facemesh  



                for x, y, z in faceMesh_landmarks:

                    face.append([x,y])

                    # ========================= # 
                    # Get the face bounding box # 
                    # ========================= #
                    #  (ref) https://github.com/google/mediapipe/issues/1737  

                    if x < cx_min:
                        cx_min = int(x)
                    
                    if y < cy_min:
                        cy_min = int(y)
                    
                    if x > cx_max:
                        cx_max = int(x)
                    if y > cy_max:
                        cy_max = int(y)

                cv2.rectangle(img, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)


                # ================================== # 
                #     Get the face area in resize    # 
                # ================================== #
                try:  
                    cropped_face = raw_img[cy_min:cy_max+1, cx_min:cx_max+1]

                    cropped_face = cv2.resize(cropped_face, dsize=(256,256), interpolation=cv2.INTER_CUBIC) # (ref) https://deep-learning-study.tistory.com/185   
                    

                    transformed_kps = self.kp_transform(face, *[cx_min, cy_min, cx_max, cy_max] )

                    onlyfaces.append(cropped_face)
                    onlyface_kps.append(transformed_kps)

                except cv2.error as e:
                    pass 

                faces_kps.append(face)

        return img, blank_img,faces_kps, onlyface_kps, onlyfaces





    def kp_transform(self, face_landmarks:list, *bbox ):

        # ==================================== # 
        # Keypoints(=landmarks) transformation # 
        # ==================================== # 
        # (ref) https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        # (ref) https://imgaug.readthedocs.io/en/latest/source/jupyter_notebooks.html

        try: 
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

        except ValueError as e: 
            # return zeros 
            return [ [0, 0] ] * 478  # face mesh 468 + iris 10



        




