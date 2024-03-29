import os 
import os.path as osp
import time 
import yaml

import cv2 
import numpy as np 
import mediapipe as mp   # by google 
from omegaconf import OmegaConf #(ref) https://majianglin2003.medium.com/python-omegaconf-a33be1b748ab

from with_mediapipe.FaceMeshModule import FaceMeshDetector





def draw_points(img, kps, radian=1, color='sapphire', thickness=-1):
    
    COLOR = {   'blue':(255,0,0), 
                'green':(0,255,0), 
                'red':(0,0,255), 
                'yellow':(0,255,255), 
                'sapphire':(255,255,0)
            }
    for x, y in kps:
        cv2.circle(img, (int(x),int(y)), radian, COLOR[color], thickness)

    return img 




#%%
def video_process(cap:cv2.VideoCapture):

    # Check if camera opened successfully
    if (cap.isOpened() == False):         
        print("Unable to read camera feed")


    try: 

        pTime = 0  # past time 
        detector = FaceMeshDetector(maxFaces=1) # only get one face 


        while True:
            ret, frame = cap.read() # read the first frame

            if not ret: 
                print(f"Reading frames {ret}")
                break 

            

            inferenced_img, landmarks_only, faces_kps, onlyface_kps, onlyfaces = detector.findFaceMesh(frame)
            

            cTime = time.time() # current time 
            process_fps = 1 / (cTime - pTime)
            pTime = cTime


            # ======================== # 
            #         Visualize        #
            # ======================== # 

            cv2.putText(frame, f"FPS: {int(process_fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.namedWindow("WebCam_view", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("WebCam_view", frame )

            cv2.namedWindow("Landmarks", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Landmarks", landmarks_only )


#            print(f"num faces: {len(faces_kps)}")

            
            for idx in range(len(faces_kps)):
                try: 
                    kp_portrait = np.zeros_like(onlyfaces[idx])
                    face_img = onlyfaces[idx].copy()

                    face_img = draw_points(face_img, onlyface_kps[idx][:468] ) # Face mesh 
                    face_img = draw_points(face_img, onlyface_kps[idx][468:], radian=2, color='red') # iris
                    kp_portrait = draw_points(kp_portrait, onlyface_kps[idx][:468] ) # Face mesh 
                    kp_portrait = draw_points(kp_portrait, onlyface_kps[idx][468:], radian=2, color='red') # iris 

                    cv2.imshow(f"ID: {idx}", face_img)
                    cv2.imshow(f"Landmakrs of ID: {idx}", kp_portrait)
                    cv2.imshow(f"Face only of ID:{idx}", onlyfaces[idx])

                except IndexError as e:
                    pass 

            key = cv2.waitKey(1)
            if key==ord('q') : # 'ESC'
                break


    finally:
        cv2.destroyAllWindows()

        # _Stop streaming
        cap.release()        




#%% 
if __name__ == '__main__':
    cfg = OmegaConf.load("cfg.yaml")
    
    data_path = cfg.Required.inputPath

    # ================ # 
    # Get video frames #
    # ================ #
    cap = cv2.VideoCapture(data_path) 
    video_fps = cap.get(cv2.CAP_PROP_FPS) 	# get default video FPS
    print(f"fps: {video_fps}")

    scaling_width = cfg.Required.displayWidth 
    scaling_height = cfg.Required.displayHeight 
    
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, scaling_width)   # (ref) https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, scaling_height)   # (ref) https://docs.opencv.org/4.5.2/dd/d43/tutorial_py_video_display.html

#    print(f"Changing resolution: {ret} ")
    print(f"width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")




    # ================ # 
    # Start processing #
    # ================ #
    video_process(cap)
