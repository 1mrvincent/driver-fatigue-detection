import math
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

STATES = ['CLOSE', 'OPEN']
TEXT_COLOR=(255, 0, 0)

# def _normalized_to_pixel_coordinates(
#     normalized_x: float, normalized_y: float, image_width: int,
#     image_height: int) :
#   """Converts normalized value pair to pixel coordinates."""

#   # Checks if the float value is between 0 and 1.
#   def is_valid_normalized_value(value: float) -> bool:
#     return (value > 0 or math.isclose(0, value)) and (value < 1 or
#                                                       math.isclose(1, value))

#   if not (is_valid_normalized_value(normalized_x) and
#           is_valid_normalized_value(normalized_y)):
#     # TODO: Draw coordinates even if it's outside of the image bounds.
#     return None
#   x_px = min(math.floor(normalized_x * image_width), image_width - 1)
#   y_px = min(math.floor(normalized_y * image_height), image_height - 1)
#   return x_px, y_px

def draw_landmarks_on_image(rgb_image, detection_result, faceDeresult, model):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  left_eye_img = cv2.imread('./left.jpg')
  right_eye_img = cv2.imread('./right.jpg')

  # Loop through the detected faces to visualize.
  # for idx in range(len(face_landmarks_list)):
  #   face_landmarks = face_landmarks_list[idx]

  #   # Draw the face landmarks.
  #   face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  #   face_landmarks_proto.landmark.extend([
  #     landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
  #   ])

  #   # solutions.drawing_utils.draw_landmarks(
  #   #     image=annotated_image,
  #   #     landmark_list=face_landmarks_proto,
  #   #     connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
  #   #     landmark_drawing_spec=None,
  #   #     connection_drawing_spec=mp.solutions.drawing_styles
  #   #     .get_default_face_mesh_contours_style())
    
  #   # solutions.drawing_utils.draw_landmarks(
  #   #     image=annotated_image,
  #   #     landmark_list=face_landmarks_proto,
  #   #     connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
  #   #     landmark_drawing_spec=None,
  #   #     connection_drawing_spec=mp.solutions.drawing_styles
  #   #     .get_default_face_mesh_contours_style())
    
   
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
      right_eye = fdetection.keypoints[0]
      left_eye = fdetection.keypoints[1]
      
      
      
      if right_eye:
          # keypoint_px = _normalized_to_pixel_coordinates(right_eye.x, right_eye.y, width, height)
          # color, thickness, radius = (0, 255, 0), 2, 2# 
          # cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
          
          origin_x = min(math.floor(right_eye.x * width), width - 1)
          origin_y = min(math.floor(right_eye.y * height), height - 1)
      
          start_point = origin_x - math.floor(bbox.width/7) , origin_y - math.floor(bbox.height/7)
          end_point = origin_x + math.floor(bbox.width/7), origin_y + math.floor(bbox.height/7)
          
          text_start_point = origin_x - math.floor(bbox.width/7) , (origin_y - math.floor(bbox.height/7))-10
          
          x = origin_x - math.floor(bbox.width/7)
          y = origin_y - math.floor(bbox.height/7)
          xw = origin_x + math.floor(bbox.width/7)
          yh = origin_y + math.floor(bbox.height/7)
          
          right_eye_img = annotated_image[y:yh, x:xw]
          try:
            cv2.imwrite('right.jpg', right_eye_img)
          except Exception as e:
            print(e)
          
          right_eye_state = predict_eye_state(model, './right.jpg')
          print(f'right eye prediction is >>> {STATES[right_eye_state]}')
          cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 2)
          cv2.putText(annotated_image, STATES[right_eye_state], text_start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

      
      
      if left_eye:
          # keypoint_px = _normalized_to_pixel_coordinates(left_eye.x, left_eye.y, width, height)
          # color, thickness, radius = (0, 255, 0), 2, 2
          # cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
          
          origin_x = min(math.floor(left_eye.x * width), width - 1)
          origin_y = min(math.floor(left_eye.y * height), height - 1)
          start_point = origin_x - math.floor(bbox.width/7) , origin_y - math.floor(bbox.height/7)
          end_point = origin_x + math.floor(bbox.width/7), origin_y + math.floor(bbox.height/7)
          
          text_start_point = origin_x - math.floor(bbox.width/7) , (origin_y - math.floor(bbox.height/7))-10
      
          
          x = origin_x - math.floor(bbox.width/7)
          y = origin_y - math.floor(bbox.height/7)
          xw = origin_x + math.floor(bbox.width/7)
          yh = origin_y + math.floor(bbox.height/7)
          
          left_eye_img = annotated_image[y:yh, x:xw]
          try:
            cv2.imwrite('left.jpg', left_eye_img)
          except Exception as e:
            print(e)
          
          left_eye_state = predict_eye_state(model, './left.jpg')
          print(f'left eye prediction is >>> {STATES[left_eye_state]}')
          cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 2)
          cv2.putText(annotated_image, STATES[left_eye_state], text_start_point, cv2.FONT_HERSHEY_SIMPLEX,0.5,TEXT_COLOR, 2)
          
      
      

  return annotated_image, left_eye_img, right_eye_img

def predict_eye_state(model, eye_img_path:str):
  
  input_details = model.get_input_details()
  output_details = model.get_output_details()
  
  image = tf.keras.preprocessing.image.load_img(eye_img_path, target_size=(20,20), grayscale=True)
  x = tf.keras.preprocessing.image.img_to_array(image)
  x = tf.expand_dims(x, 0)
  model.set_tensor(input_details[0]['index'], x)
  model.invoke()
  prediction = model.get_tensor(output_details[0]['index'])
  prediction = tf.concat(prediction, axis=0)
  prediction = tf.argmax(prediction, axis=1)
  return prediction[0]

