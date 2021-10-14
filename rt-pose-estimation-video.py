import cv2
import mediapipe as mp
import av
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

container = av.open("C:\\Users\\cleme\\Documents\\EPFL\\Master\\MA-3\\sensor\\data\\cam1_911222060374_record_30_09_2021_1359_49.avi")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  for frame in container.decode(video=0):
    frame = frame.reformat(frame.width, frame.height, 'rgb24')
    image = frame.to_ndarray()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('',image)
    # if cv2.waitKey(5) & 0xFF == 27:
    #   break

    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(image)

    if not results.pose_landmarks:
      continue
    print(
        f'Left wrist coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height})'
    )


    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

