from ntpath import join
import cv2
import mediapipe as mp
import av
import numpy as np
import matplotlib.pyplot as plt

from video_tracking.filter_bank import hampel_filter, identity_filter

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

mp_pose = mp.solutions.pose

landmarks_list = [mp_pose.PoseLandmark.LEFT_WRIST,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_SHOULDER]

additional_landmarks_list = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.LEFT_EAR,
  mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP]

landmarks_list += additional_landmarks_list

raw_x_values = {landmark: [] for landmark in landmarks_list}
filtered_x_values = {landmark: [] for landmark in landmarks_list}
raw_y_values = {landmark: [] for landmark in landmarks_list}
filtered_y_values = {landmark: [] for landmark in landmarks_list}
raw_z_values = {landmark: [] for landmark in landmarks_list}
filtered_z_values = {landmark: [] for landmark in landmarks_list}

#x_filter = lambda raw_x_values, current_x: hampel_filter(raw_x_values, current_x)
#y_filter = lambda raw_y_values, current_y: hampel_filter(raw_y_values, current_y)
x_filter = lambda raw_x_values, current_x: identity_filter(current_x)
y_filter = lambda raw_y_values, current_y: identity_filter(current_y)
z_filter = lambda raw_z_values, current_z: hampel_filter(raw_z_values, current_z, window_size=20, window_offset=-10)

# Bones:
# left_shoulder - right_shoulder
bone_list = [[mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW], [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER],
  [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER], [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW],
  [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER]]

additional_bones_list = [[mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
  [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP],
  [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP]]

bone_list += additional_bones_list

raw_bones = [[0, 0, 0, 0, 0, 0] for _ in bone_list]
filtered_bones = [[0, 0, 0, 0, 0, 0] for _ in bone_list]

raw_joint_positions = {}
filtered_joint_positions = {}

def estimate_pose(pose, rgb_image, depth_image, depth_scale=1):
  image_height, image_width, _ = rgb_image.shape
  results = pose.process(rgb_image)

  if results.pose_landmarks == None:
    print("No pose found, check if someone is in the camera field!")

  screen_depth_scale = 100

  for landmark in landmarks_list:
      coord = results.pose_landmarks.landmark[landmark]
      x = min(int(coord.x * image_width), 640-1)
      y = min(int(coord.y * image_height), 480-1)
      z = screen_depth_scale * depth_scale * depth_image[y, x]

      raw_joint_positions[landmark] = [x, y, z]

      if raw_x_values[landmark]:
        filtered_x = x_filter(raw_x_values[landmark], x)
      else:
        filtered_x = x

      # WARNING: only append to values after filtering!
      raw_x_values[landmark].append(x)

      filtered_x_values[landmark].append(filtered_x) # TODO necessary?

      if raw_y_values[landmark]:
        filtered_y = y_filter(raw_y_values[landmark], y)
      else:
        filtered_y = y

      # WARNING: only append to values after filtering!
      raw_y_values[landmark].append(y)

      filtered_y_values[landmark].append(filtered_y) # TODO necessary?

      if raw_z_values[landmark]:
        filtered_z = z_filter(raw_z_values[landmark], z)
      else:
        filtered_z = z

      # WARNING: only append to values after filtering!
      raw_z_values[landmark].append(z)

      filtered_z_values[landmark].append(filtered_z) # TODO necessary?

      filtered_joint_positions[landmark] = [filtered_x, filtered_y, filtered_z]

  # Flip the image horizontally for a selfie-view display.
  #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

  for i, bone in enumerate(bone_list):
    x1, y1, z1 = raw_joint_positions[bone[0]]
    x2, y2, z2 = raw_joint_positions[bone[1]]
    raw_bones[i] = [x1, y1, z1, x2, y2, z2]

    x1, y1, z1 = filtered_joint_positions[bone[0]]
    x2, y2, z2 = filtered_joint_positions[bone[1]]
    filtered_bones[i] = [x1, y1, z1, x2, y2, z2]

  return raw_joint_positions, filtered_joint_positions, raw_bones, filtered_bones, results