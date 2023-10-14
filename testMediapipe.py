import os
import time
import mediapipe as mp
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from helper_funcs import draw_landmarks_on_image
import tensorflow as tf
import numpy as np


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

TEXT_COLOR=(255, 0, 0)

SLEEP_THRESHOLD = 8

model = tf.lite.Interpreter(model_path='./kaggle_best_model__final_284k.tflite')
model.allocate_tensors()

STATES = ['CLOSE', 'OPEN']


def predict_eye_state(model):
  
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    left_image = tf.keras.preprocessing.image.load_img('./out/left.jpg', target_size=(20,20), grayscale=True)
    right_image = tf.keras.preprocessing.image.load_img('./out/right.jpg', target_size=(20,20), grayscale=True)
    
    lx = tf.keras.preprocessing.image.img_to_array(left_image)
    lx = tf.expand_dims(lx, 0)
    rx = tf.keras.preprocessing.image.img_to_array(right_image)
    rx = tf.expand_dims(rx, 0)
    
    model.set_tensor(input_details[0]['index'], lx)
    model.invoke()
    lprediction = model.get_tensor(output_details[0]['index'])
    lprediction = tf.concat(lprediction, axis=0)
    lprediction = tf.argmax(lprediction, axis=1)
    
    model.set_tensor(input_details[0]['index'], rx )
    model.invoke()
    rprediction = model.get_tensor(output_details[0]['index'])
    rprediction = tf.concat(rprediction, axis=0)
    rprediction = tf.argmax(rprediction, axis=1)
    
    prediction = '{:.2f}'.format((rprediction[0]+lprediction[0])/2)
    
    if float(prediction) >= 0.5:
        print(f'xxxxxxxxxxxxxxx-==================== open left>> {lprediction} ~~~~~ right >>{rprediction} ~~~~~~~~~~~ {prediction}')
        return 1
    else:
        print(f'xxxxxxxxxxxxxxx-==================== close left >> {lprediction} ~~~~~ right>>{rprediction}~~~~~~~~~~~ {prediction}')
        return 0

try:
        
    count = 0
    while True:
        
        # Read camera frames
        success, frame = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue    # use break if reading from a video file 
   
        h, w, c = frame.shape
        
        # flip the video vertically and change the BGR to RGB
        frame = cv2.flip(frame, 1)

        fbase_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
        foptions = vision.FaceDetectorOptions(base_options=fbase_options)
        fdetector = vision.FaceDetector.create_from_options(foptions)
        
        # show webcame frames
        # cv2.imshow("Webcam", gray_image)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        
        face_detection_result = fdetector.detect(mp_image)
        
        annotated_image, left_eye, right_eye = draw_landmarks_on_image(mp_image.numpy_view(), face_detection_result)
    
                        
         
        # cv2.imshow(mat=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY), winname='image')  
        
        print(f'shape of left eye {left_eye.shape}')
        print(f'shape of right eye {right_eye.shape}')
        
        # cv2.imshow(mat=left_eye, winname='left eye')
        # cv2.imshow(mat=right_eye,  winname='right eye')
    
        
        prediction = predict_eye_state(model)
        print(STATES[prediction])
        if prediction == 1:
            count = count-1 if count != 0 else 0
        else:
            if count > 20:
                count=0
            count = count+1
        
        print(count)
        if count >= SLEEP_THRESHOLD :
            cv2.putText(annotated_image, "Sleep detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX,0.5,TEXT_COLOR, 2)
            
        cv2.imshow(mat=annotated_image, winname='image') 
       
        
        
        # press q to exit  
        if cv2.waitKey(1) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
        
        time.sleep(1/4)
except Exception as e :
    print(e)
    

# Finally release back the camera resources 
cap.release()

