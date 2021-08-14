

import sys 
import os 
import os.path as osp
import time 
import yaml

import cv2 
import numpy as np 
import mediapipe as mp   # by google 
from tqdm import tqdm 
from omegaconf import OmegaConf #(ref) https://majianglin2003.medium.com/python-omegaconf-a33be1b748ab

from utils.FaceMeshModule import FaceMeshDetector












#%%
def video_process(cap:cv2.VideoCapture):

    # Check if camera opened successfully
    if (cap.isOpened() == False):         
        print("Unable to read camera feed")


    try: 

        pTime = 0  # past time 
        detector = FaceMeshDetector(maxFaces=1)

        while True:
            ret, frame = cap.read() # read the first frame

            if not ret: 
                print(f"Reading frames {ret}")
                break 

            

            inferenced_img, blank, faces = detector.findFaceMesh(frame)
            



            cTime = time.time() # current time 
            process_fps = 1 / (cTime - pTime)
            pTime = cTime


            cv2.putText(frame, f"FPS: {int(process_fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.namedWindow("WebCam_view", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("WebCam_view", frame )

            cv2.namedWindow("Landmarks", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Landmarks", blank )

            
            key = cv2.waitKey(1)

            
            if key == 27 : # 'ESC'
                break





    finally:
        cv2.destroyAllWindows()

        # _Stop streaming
        cap.release()        









#%% 
if __name__ == '__main__':
    cfg = OmegaConf.load("cfg.yaml")
    
    data_dir = cfg.Required.inputPath
    video_list = sorted(os.listdir(data_dir))    
    data_path = osp.join(data_dir, video_list[0])

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