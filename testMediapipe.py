import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from helper_funcs import draw_landmarks_on_image
import tensorflow as tf


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connection = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
mp_face_dection = mp.solutions.face_detection


cap = cv2.VideoCapture(1)
# set the setting of the webcam
cap.set(3, 640)    # width
cap.set(4, 420)    # height
cap.set(10, 100)   # brightness


model = tf.lite.Interpreter(model_path='./kaggle_best_model__final-2.tflite')
model.allocate_tensors()



while True:
    # Read camera frames
    success, frame = cap.read()
    if not success:
        print('Ignoring empty camera frame.')
        continue    # use break if reading from a video file 
      
    print(f'image shape >>> {frame.shape}')
    h, w, c = frame.shape
    
    # flip the video vertically and change the BGR to RGB
    frame = cv2.flip(frame, 1)
    print('frame is flipped')
    
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=True,
                                        running_mode=RunningMode.IMAGE,
                                        num_faces=1)
    
    

    fbase_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    foptions = vision.FaceDetectorOptions(base_options=fbase_options)
    fdetector = vision.FaceDetector.create_from_options(foptions)
    
    # show webcame frames
    # cv2.imshow("Webcam", gray_image)
    
    landmarker = vision.FaceLandmarker.create_from_options(options)
        
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result =landmarker.detect(mp_image)
    
    fdetection_result = fdetector.detect(mp_image)
    
    annotated_image, left_eye, right_eye = draw_landmarks_on_image(mp_image.numpy_view(), result, fdetection_result, model)
    try:
                        
         
        # cv2.imshow(mat=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY), winname='image')  
        
        print(f'shape of left eye {left_eye.shape}')
        print(f'shape of right eye {right_eye.shape}')
        
        # cv2.imshow(mat=left_eye, winname='left eye')
        # cv2.imshow(mat=right_eye,  winname='right eye')
        
        cv2.imshow(mat=annotated_image, winname='image') 
        
        
        # press q to exit  
        if cv2.waitKey(1) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
    except Exception as e :
        print(e)
    

# Finally release back the camera resources 
cap.release()

