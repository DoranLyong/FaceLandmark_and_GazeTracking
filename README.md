# 468 Face Landmarks tutorial 

#### 1. Install requirements 

```bash 
~$ pip install -r -U requirements.txt
```



For the iris example, download ```iris_landmark.tflite``` 

```bash
~$ wget -P models https://github.com/google/mediapipe/raw/master/mediapipe/modules/iris_landmark/iris_landmark.tflite

```





#### 2. Run the code 

```bash
~$ python FaceMesh_with_rough_pupil.py  # Mediapipe face mesh & OpenCV usage eye pupil 

~$ python FaceMesh_with_iris.py # Mediapipe face mesh & iris 
```





***
### Reference 
[1] [468 Face Landmars, CVZONE](https://www.computervision.zone/courses/468-face-landmarks/?ld-registered=true) / <br/>
[2] [Detect 468 Face Landmarks in Real-time | OpenCV Python, youtube](https://youtu.be/V9bzew8A1tc) /  <br/>
[3] [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html) / <br/>
[4] [ARFaceAnchor, apple developer](https://developer.apple.com/videos/play/tech-talks/601/) / <br/>
[5] [Malla Facial](https://omes-va.com/malla-facial-mediapipe-python/) / <br/>
[6] [Real-time Facial Performance Capture with iPhone X](https://github.com/johnjcsmith/iPhoneMoCapiOS) / <br/>
[7-1] [FaceMesh: Detecting Key Points on Faces in Real Time](https://medium.com/axinc-ai/facemesh-detecting-key-points-on-faces-in-real-time-977c03f1bab) / 얼굴 매쉬 이미지만 취득하려다가 여기까지 왔네... <br/>[7-2] [ailia-models, github](https://github.com/axinc-ai/ailia-models/tree/master/face_recognition/facemesh) / 위 포스트의 실습코드인데, 여기 레포지토리에 기타 실행해보면 좋을 모델이 많이 있다 <br/>
[[8] [Face Mesh| 468 facial landmark detection | mediapipe, youtube](https://youtu.be/7WPdEajSL6c) / <br/>
[9] [Attention Mesh: High-fidelity Face Mesh Prediction in Real-time](https://www.arxiv-vanity.com/papers/2006.10962/) / 논문 설명 <br/>
[10] [Face Mesh, ailia-models, github](https://github.com/axinc-ai/ailia-models/tree/master/face_recognition/facemesh) / <br/>



###### Pupil detection

[11] [Gaze Tracking, github](https://github.com/antoinelame/GazeTracking) / iris 위치 추출을 위한 예시 참고  <br/>
[12] [Real-time Pupil Tracking from monocular Video, github](https://github.com/cedriclmenard/irislandmarks.pytorch) / Google Mediapipe의  iris detection 인데 이건 PyTorch로 구현 됨 <br/>[13] [MediaPipe Iris: 실시간 홍채추적 및 깊이측정](https://brunch.co.kr/@synabreu/93) / <br/>
[14] [How use Mediapipe API for Python Iris Depth?](https://github.com/google/mediapipe/issues/2254) / 파이썬 버전으로 된게 없어 이것저것 찾아봄. 대체로 그냥 옵션 추가후 수동 빌드를 통해 해결한는 것 같다? <br/>
[15]  [mediapipeDemos, github](https://github.com/Rassibassi/mediapipeDemos)  / 위의 링크에서 찾았다. 드디어! 이거 참고해서 새로 만들어야지 <br/>
[16] [MediaPipe Iris: Detecting Key Points in the Eye, medium](https://medium.com/axinc-ai/mediapipe-iris-detecting-key-points-in-the-eye-637f5c1e728e) /  <br/>

