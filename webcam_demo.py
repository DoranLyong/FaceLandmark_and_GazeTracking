import sys 
import os 
import os.path as osp
import time 
import yaml

import cv2 
import numpy as np 
import mediapipe as mp   # by google  
from omegaconf import OmegaConf #(ref) https://majianglin2003.medium.com/python-omegaconf-a33be1b748ab



def main(): 
    #Opening OpenCV stream
    stream = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while stream.isOpened():
            ret,img = stream.read()
            print(img.shape)



            cv2.imshow('Human Pose Estimation', img)
            key = cv2.waitKey(1)
            if key==ord('q'):
                    break
    stream.release()
    cv2.destroyAllWindows()




#%% 
if __name__ == '__main__':
    main() 
