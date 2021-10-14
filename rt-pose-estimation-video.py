import cv2
import mediapipe as mp
import av

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

container = av.open("D:/sensorimotor-data/recording1.avi")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  for frame in container.decode(video=0):
    image = frame.to_ndarray()
    cv2.imshow(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #results = pose.process(image)

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #mp_drawing.draw_landmarks(
        #image,
        #results.pose_landmarks,
        #mp_pose.POSE_CONNECTIONS,
        #landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    # Plot pose world landmarks.
    #mp_drawing.plot_landmarks(
        #results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    #if cv2.waitKey(5) & 0xFF == 27:
      #break