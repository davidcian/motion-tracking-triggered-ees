from ntpath import join
import sys
import random
import argparse
import os
import csv
import time

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QPushButton, QComboBox, QMainWindow
from PySide6.QtCore import Slot
from PySide6.QtGui import QVector3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from video_tracking.rt_pose_estimation import estimate_pose, bone_list, landmarks_list

import numpy as np
import mediapipe as mp
import cv2

from image_data_providers import PyRealSenseCameraProvider, PyRealSenseVideoProvider, WebcamProvider

from opensim_tools import calculate_angle, AngleTraj, path_planning

from coordinate_display import CoordinatePlotWidget
from implant_display import ImplantWidget

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pen1 = pg.mkPen(color='r', width=2)
pen2 = pg.mkPen(color='g', width=2)
pen3 = pg.mkPen(color='b', width=2)

class Skeleton():
  def __init__(self, joint_color, bone_color):
    self.joint_positions = []
    self.bones = []
    self.joint_color = joint_color
    self.bone_color = bone_color

    self.bone_item_positions = []
    self.bone_items = []

    for i in range(len(bone_list)):
      self.bone_item_positions.append(np.array([[0, 0, 0], [0, 0, 0]]))
      self.bone_items.append(gl.GLLinePlotItem(pos=self.bone_item_positions[i], width=1))

class MyWidget(QtWidgets.QWidget):
  def __init__(self, pose, image_data_provider):
    super().__init__()

    self.pose = pose
    self.image_data_provider = image_data_provider

    self.depth_scale = self.image_data_provider.depth_scale

    self.coordinate_plot_widget = CoordinatePlotWidget()

    self.implant_widget = ImplantWidget()

    self.current_frame = 0

    self.frame_indices = list(range(self.current_frame))

    self.layout = QtWidgets.QVBoxLayout(self)

    self.skeletons = {'raw': Skeleton([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]), 'filtered': Skeleton([0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0])}

    ###

    w = gl.GLViewWidget()

    self.pos = np.array([1, 1, 3])
    self.sp2 = gl.GLScatterPlotItem(pos=self.pos)

    # Draw a grid as the floor
    g = gl.GLGridItem(size=QVector3D(1000, 1000, 1000))
    g.setSpacing(spacing=QVector3D(100, 100, 100))
    w.addItem(g)

    # Draw the axes
    axes = gl.GLAxisItem(size=QVector3D(2000, 2000, 2000))
    w.addItem(axes)

    w.addItem(self.sp2)

    ###

    # Draw bones
    for skeleton in self.skeletons.values():
      for bone_item in skeleton.bone_items:
        w.addItem(bone_item)

    self.layout.addWidget(w)

    self.timer = QtCore.QTimer(self)
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), lambda: self.update_plot_data())
    update_interval = 100
    self.timer.start(update_interval)

    show_coordinate_plot_button = QPushButton("Show coordinate plots")
    show_coordinate_plot_button.clicked.connect(self.show_coordinate_plots)
    self.layout.addWidget(show_coordinate_plot_button)

    show_implant_widget_button = QPushButton("Show implant stimulation")
    show_implant_widget_button.clicked.connect(self.show_implant_stimulation)
    self.layout.addWidget(show_implant_widget_button)

    self.monte = True
    self.descend = False
    self.counter = 0
    self.stage = 'Down'

  @Slot()
  def show_implant_stimulation(self):
    self.implant_widget.show()

  @Slot()
  def show_coordinate_plots(self):
    self.coordinate_plot_widget.show()

  def update_plot_data(self):
    rgb_image, depth_image = self.image_data_provider.retrieve_rgb_depth_image()

    # Color image of the depth
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

    cv2.imshow('RealSense', depth_colormap)

    # Estimate the pose with MediaPipe
    raw_joint_positions, filtered_joint_positions, raw_bones, filtered_bones, results = estimate_pose(self.pose, rgb_image, depth_image, self.depth_scale)
    self.results = results
    self.skeletons['raw'].joint_positions, self.skeletons['raw'].bones = raw_joint_positions, raw_bones
    self.skeletons['filtered'].joint_positions, self.skeletons['filtered'].bones = filtered_joint_positions, filtered_bones

    # Update angle_traj_widget by updating angle and time values:
    landmarks = self.results.pose_world_landmarks.landmark

    # Get coordinates
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    angle = calculate_angle(shoulder, elbow, wrist)
    angle = abs(angle - 180)
    if hasattr(self, 'angle_traj_widget'):
      delta_t = time.time() - self.angle_traj_widget.time_begin
      self.angle_traj_widget.update_values(delta_t,angle)
      print(delta_t)
      print(angle)

    if angle < 30 and self.monte == True and self.descend != True:
      self.stage = "down"
      self.monte = False
      self.descend = True
    if angle > 120 and self.stage =='down' and self.descend == True and self.monte == False :
      self.stage="up"
      self.descend = False
      self.monte = True
      self.counter +=1
      print(self.counter)

    ###

    mp_drawing.draw_landmarks(
      rgb_image,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    ###

    # Render curl counter
    # Setup status box
    cv2.rectangle(rgb_image, (0,0), (225,78), (0,0,255), -1)
    cv2.line(rgb_image,pt1=(100,0), pt2=(100,78), color=(255,255,255), thickness=2)
    cv2.line(rgb_image,pt1=(225,0), pt2=(225,78), color=(255,255,255), thickness=2)
    cv2.line(rgb_image,pt1=(0,78), pt2=(225,78), color=(255,255,255), thickness=2)

    # Rep data
    cv2.putText(rgb_image, 'REPS', (5,31), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(rgb_image, str(self.counter), (25,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(rgb_image, 'STAGE', (110,31), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(rgb_image, self.stage, (110,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    ###

    cv2.imshow('MediaPipe Pose', rgb_image)

    self.current_frame += 1

    #self.frame_indices = self.frame_indices[1:]
    #self.frame_indices.append(self.frame_indices[-1] + 1)
    self.frame_indices.append(self.current_frame)

    selected_joint = self.coordinate_plot_widget.selected_joint

    x, y, z = raw_joint_positions[selected_joint]
    filtered_x, filtered_y, filtered_z = filtered_joint_positions[selected_joint]

    features_update = {'x_val': x, 'y_val': y, 'z_val': z, 'filtered_z_val': filtered_z}
    self.coordinate_plot_widget.update(self.frame_indices, features_update)

    ###

    skeletons_pos = []
    skeletons_color = []

    for skeleton in self.skeletons.values():
      pos = np.empty([len(skeleton.joint_positions), 3])
      color = np.array([skeleton.joint_color for _ in range(len(skeleton.joint_positions))])
      idx = 0
      for joint_name, joint_position in skeleton.joint_positions.items():
        #raw_pos[idx] = joint_position
        pos[idx, 0] = joint_position[0]
        pos[idx, 1] = joint_position[2]
        pos[idx, 2] = joint_position[1]
        pos[idx, 2] = -pos[idx, 2]
        pos[idx, 2] += 400
        idx += 1

      skeletons_pos.append(pos)
      skeletons_color.append(color)

    all_pos = np.vstack(skeletons_pos)
    all_color = np.vstack(skeletons_color)

    self.sp2.setData(pos=all_pos, color=all_color)

    ###

    for skeleton in self.skeletons.values():
      # Plot raw (red) and filtered data (blue) data bones
      for i, bone in enumerate(skeleton.bones):
        x1, z1, y1, x2, z2, y2 = bone
        z1, z2 = -z1, -z2
        z1, z2 = z1 + 400, z2 + 400
        skeleton.bone_item_positions[i] = np.array([[x1, y1, z1], [x2, y2, z2]])
        skeleton.bone_items[i].setData(pos=skeleton.bone_item_positions[i], color=skeleton.bone_color)

  def get_pos(self):
    self.update_plot_data()

    landmarks = self.results.pose_world_landmarks.landmark

    # Get coordinates
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    return wrist, shoulder

  def set_trajectory(self,angle_traj, time_traj):
    self.angle_traj_widget = AngleTraj(angle_traj, time_traj)
    self.angle_traj_widget.setWindowTitle('Trajectory window')
    self.angle_traj_widget.time_begin = time.time()
    self.angle_traj_widget.resize(800, 400)
    self.angle_traj_widget.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--source", default='video')
  parser.add_argument("-d", "--dir", default="C:\\Users\\cleme\\Documents\\EPFL\\Master\\MA-3\\sensor\\data\\")
  parser.add_argument("-z", "--depth-file")
  parser.add_argument("-w", "--with-depth")
  args = parser.parse_args()

  try:
    if args.source == 'video':
      image_data_provider = PyRealSenseVideoProvider(file_path=os.path.join(args.dir, args.depth_file))
    elif args.source == 'pyrealsense':
      image_data_provider = PyRealSenseCameraProvider()
    elif args.source == 'webcam':
      image_data_provider = WebcamProvider()

    with mp_pose.Pose(static_image_mode=False,
      model_complexity=2,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:

      app = QtWidgets.QApplication([])

      widget = MyWidget(pose, image_data_provider)
      widget.resize(800, 600)
      #widget.show()

      main_window = QMainWindow()
      main_window.setCentralWidget(widget)
      main_window.show()

      wrist, shoulder = widget.get_pos()

      wrist_pos_i = [0.2, round(0.4 - wrist[1], 1), round(-wrist[0], 1)]
      wrist_pos_f = [0.1, round(0.4 - shoulder[1], 1), round(-shoulder[0], 1)]
      # landmark_init = results.pose_landmarks
      print(wrist_pos_i)
      print(wrist_pos_f)
      print("main call")

      angle_traj, time_traj = path_planning(wrist_pos_i, wrist_pos_f)
      print(angle_traj)
      print(angle_traj.shape)
      print(angle_traj[:,1].shape)
      print(time_traj)
      angle_traj_2, time_traj_2 = path_planning(wrist_pos_f, wrist_pos_i)
      angle_traj = np.concatenate([angle_traj,angle_traj_2])
      time_traj_2 = time_traj_2 + np.max(time_traj)
      time_traj = np.concatenate([time_traj,time_traj_2])
      print(angle_traj)
      print(angle_traj.shape)
      print(angle_traj[:,1].shape)
      print(time_traj)
      print(time_traj)
      for i in range(10):
          angle_traj = np.concatenate([angle_traj,angle_traj])
          time_traj_add = time_traj + np.max(time_traj)
          time_traj = np.concatenate([time_traj,time_traj_add])
      print("FINAL")
      print(angle_traj)
      print(angle_traj.shape)
      print(angle_traj[:,1].shape)
      print(time_traj)
      print(time_traj)
      widget.set_trajectory(angle_traj[:,1], time_traj)

      sys.exit(app.exec())
  finally:
    image_data_provider.stop()