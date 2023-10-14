import math
import os
import mediapipe as mp
import cv2
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

TEXT_COLOR=(255, 0, 0)

def draw_landmarks_on_image(rgb_image, faceDeresult):
  
  annotated_image = np.copy(rgb_image)
  left_eye_img = cv2.imread('./out/left.jpg')
  right_eye_img = cv2.imread('./out/right.jpg')

    
   
  height, width, _ = rgb_image.shape
#   annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY)

  
  for fdetection in faceDeresult.detections:
      # Draw bounding_box
      print(f'locccc >>>>>{str(fdetection.bounding_box)}')
      bbox = fdetection.bounding_box
      # start_point = bbox.origin_x, bbox.origin_y
      # end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
      # cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
      
      # print(type(fdetection))


  #   for keypoint in fdetection.keypoints: # mp_face_dection.get_key_point(fdetection.keypoints, mp_face_dection.FaceKeyPoint.RIGHT_EYE):
      right_eye = fdetection.keypoints[1]
      left_eye = fdetection.keypoints[0]
      
      
      
      if right_eye:
          # color, thickness, radius = (0, 255, 0), 2, 2# 
          # cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
          
          origin_x = min(math.floor(right_eye.x * width), width - 1)
          origin_y = min(math.floor(right_eye.y * height), height - 1)
      
          start_point = origin_x - math.floor(bbox.width/6) , origin_y - math.floor(bbox.height/8)
          end_point = origin_x + math.floor(bbox.width/6), origin_y + math.floor(bbox.height/8)
          
          text_start_point = origin_x - math.floor(bbox.width/6) , (origin_y - math.floor(bbox.height/6))-10
          
          x = origin_x - math.floor(bbox.width/6)
          y = origin_y - math.floor(bbox.height/6)
          xw = origin_x + math.floor(bbox.width/6)
          yh = origin_y + math.floor(bbox.height/6)
          
          right_eye_img = annotated_image[y:yh, x:xw]
          try:
            cv2.imwrite('./out/right.jpg', right_eye_img)
          except Exception as e:
            print(e)
          
          # right_eye_state = predict_eye_state(model, './out/right.jpg')
          # print(f'right eye prediction is >>> {STATES[right_eye_state]}')
          cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 2)
          # cv2.putText(annotated_image, STATES[right_eye_state], text_start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

      
      
      if left_eye:
          # keypoint_px = _normalized_to_pixel_coordinates(left_eye.x, left_eye.y, width, height)
          # color, thickness, radius = (0, 255, 0), 2, 2
          # cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
          
          origin_x = min(math.floor(left_eye.x * width), width - 1)
          origin_y = min(math.floor(left_eye.y * height), height - 1)
          start_point = origin_x - math.floor(bbox.width/6) , origin_y - math.floor(bbox.height/8)
          end_point = origin_x + math.floor(bbox.width/6), origin_y + math.floor(bbox.height/8)
          
          text_start_point = origin_x - math.floor(bbox.width/6) , (origin_y - math.floor(bbox.height/6))-10
      
          
          x = origin_x - math.floor(bbox.width/6)
          y = origin_y - math.floor(bbox.height/6)
          xw = origin_x + math.floor(bbox.width/6)
          yh = origin_y + math.floor(bbox.height/6)
          
          left_eye_img = annotated_image[y:yh, x:xw]
          try:
            cv2.imwrite('./out/left.jpg', left_eye_img)
          except Exception as e:
            print(e)
          
          # left_eye_state = predict_eye_state(model, './out/left.jpg')
          # print(f'left eye prediction is >>> {STATES[left_eye_state]}')
          cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 2)
          # cv2.putText(annotated_image, STATES[left_eye_state], text_start_point, cv2.FONT_HERSHEY_SIMPLEX,0.5,TEXT_COLOR, 2)
          

  return annotated_image, left_eye_img, right_eye_img



