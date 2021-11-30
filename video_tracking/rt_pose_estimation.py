from ntpath import join
import cv2
import mediapipe as mp
import av
import numpy as np
import matplotlib.pyplot as plt

from video_tracking.filter_bank import filter_z

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

landmarks_coord = []

landmarks_list = [mp_pose.PoseLandmark.LEFT_WRIST,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_SHOULDER]

depth_values = []

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.set_title("Skeleton of patient")

# Bones:
# left_shoulder - right_shoulder
bone_list = [[mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW], [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER],
  [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER], [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW],
  [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER]]

bones = [[0, 0, 0, 0, 0, 0] for _ in bone_list]

joint_positions = {}

def estimate_pose(pose, color_frame, depth_frame, depth_scale, current_frame):
  depth_image_1 = np.asanyarray(depth_frame.get_data())
  color_image_1 = np.asanyarray(color_frame.get_data())

  depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.05), cv2.COLORMAP_JET)

  #frame = frame.reformat(frame.width, frame.height, 'rgb24')
  #image = frame.to_ndarray()

  # Convert the BGR image to RGB before processing.
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #image = cv2.cvtColor(color_image_1,cv2.COLOR_BGR2RGB)
  image = color_image_1

  image_height, image_width, _ = image.shape
  results = pose.process(image)

  for landmark in landmarks_list:
      coord = results.pose_landmarks.landmark[landmark]
      x = min(int(coord.x * image_width), 640-1)
      y = min(int(coord.y * image_height), 480-1)
      depth_z = depth_scale * depth_image_1[y,x]
      landmarks_coord.append([current_frame,landmark,coord.x * image_width,coord.y * image_height,coord.z,depth_z])

      joint_positions[landmark] = [x, y, depth_z]

  mp_drawing.draw_landmarks(
      image,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

  # Flip the image horizontally for a selfie-view display.
  #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
  coord = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
  x = min(int(coord.x * image_width), 640-1)
  y = min(int(coord.y * image_height), 480-1)
  depth_z = depth_scale * depth_image_1[y,x]

  cv2.imshow('RealSense', depth_colormap_1)
  cv2.imshow('MediaPipe Pose', image)

  if current_frame > 1:
    filtered_z = filter_z(depth_values, depth_z)
  else:
    filtered_z = depth_z 

  depth_values.append(filtered_z)

  # Draw the depth value over time
  plt.title("Depth over time")
  plt.scatter(current_frame, depth_z, c='b')
  plt.scatter(current_frame, filtered_z, c='r')

  # Draw the skeleton over time
  #ax.cla()
  #ax.scatter(x, y, filtered_z, c='r')
  #for joint_name, joint_position in joint_positions.items():
    #x, y, z = joint_position
    #ax.scatter(x, y, z, c='r')

  for i, bone in enumerate(bone_list):
    x1, y1, z1 = joint_positions[bone[0]]
    x2, y2, z2 = joint_positions[bone[1]]
    bones[i] = [x1, y1, z1, x2, y2, z2]

  return x, y, depth_z, joint_positions, bones