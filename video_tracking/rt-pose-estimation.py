import cv2
import mediapipe as mp
import av
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

import argparse

import os

from filter_bank import filter_z

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", default='video')
parser.add_argument("-d", "--dir", default="C:\\Users\\cleme\\Documents\\EPFL\\Master\\MA-3\\sensor\\data\\")
parser.add_argument("-f", "--file", default='cam1_911222060374_record_30_09_2021_1404_05.avi')
parser.add_argument("-z", "--depth-file")
parser.add_argument("-w", "--with-depth")
args = parser.parse_args()

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

landmarks_coord = []

landmarks_list = [mp_pose.PoseLandmark.LEFT_WRIST,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_SHOULDER]

pipeline_1 = rs.pipeline()
config_1 = rs.config()

if args.source == 'video':
  rs.config.enable_device_from_file(config_1, os.path.join(args.dir, args.depth_file))
elif args.source == 'live':
  ctx = rs.context()
  devices = ctx.query_devices()
  config_1.enable_device(str(devices[0].get_info(rs.camera_info.serial_number)))
  config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

if args.with_depth == 'true':
  config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming from camera
profile = pipeline_1.start(config_1)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

depth_values = []

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.set_title("Skeleton of patient")

# Bones:
# left_shoulder - right_shoulder
bone_list = [[mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ELBOW], [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER],
  [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER], [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW],
  [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER]]

joint_positions = {}

with mp_pose.Pose(static_image_mode=False,
  model_complexity=2,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as pose:

  current_frame = 1

  try:
    while True:
      # to be check: right x,y and z
      frames_1 = pipeline_1.wait_for_frames()
      depth_frame_1 = frames_1.get_depth_frame()
      color_frame_1 = frames_1.get_color_frame()
      if not depth_frame_1 or not color_frame_1:
          continue
      depth_image_1 = np.asanyarray(depth_frame_1.get_data())
      color_image_1 = np.asanyarray(color_frame_1.get_data())

      depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.05), cv2.COLORMAP_JET)

      if cv2.waitKey(25)==113: #q pressed
              break


      #frame = frame.reformat(frame.width, frame.height, 'rgb24')
      #image = frame.to_ndarray()

      # Convert the BGR image to RGB before processing.
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      #image = cv2.cvtColor(color_image_1,cv2.COLOR_BGR2RGB)
      image = color_image_1

      image_height, image_width, _ = image.shape
      results = pose.process(image)

      if not results.pose_landmarks:
        continue

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

      cv2.putText(image,'x'+str(x)+'y'+str(y)+'depth'+str(depth_z),
          bottomLeftCornerOfText,
          font,
          fontScale,
          fontColor,
          lineType)
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

      #for bone in bone_list:
        #x1, y1, z1 = joint_positions[bone[0]]
        #x2, y2, z2 = joint_positions[bone[1]]
        #ax.plot([x1, x2], [y1, y2], [z1, z2], c='b')

      plt.pause(0.05)
      current_frame += 1
      
      if cv2.waitKey(5) & 0xFF == 27:
        break

    #ax.show()
    plt.show()

  # to do: stop process at the end of video
  finally:
      pipeline_1.stop()
      np.savetxt('./output/Landmarks_coordinates_'+str(file_name)+'.csv',landmarks_coord,delimiter=',')