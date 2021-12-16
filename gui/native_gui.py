from ntpath import join
import sys
import random
import argparse
import os
import csv
import time

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QPushButton, QComboBox, QMainWindow, QMenu, QDockWidget
from PySide6.QtCore import Slot
from PySide6.QtGui import QAction
import pyqtgraph as pg

import numpy as np
import mediapipe as mp
import cv2

from image_data_providers import PyRealSenseCameraProvider, PyRealSenseVideoProvider, WebcamProvider
from video_tracking.rt_pose_estimation import estimate_pose

from opensim_tools import calculate_angle, AngleTraj, path_planning

from skeleton_display import SkeletonWidget
from implant_display import ImplantWidget
from coordinate_display import CoordinatePlotWidget

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pen1 = pg.mkPen(color='r', width=2)
pen2 = pg.mkPen(color='g', width=2)
pen3 = pg.mkPen(color='b', width=2)

class MainWindow(QMainWindow):
  def __init__(self, pose, image_data_provider):
    super().__init__()

    self.setGeometry(100, 100, 500, 500)

    self.pose = pose
    self.image_data_provider = image_data_provider

    self.current_frame = 0
    self.frame_indices = list(range(self.current_frame))

    self.menu_bar = self.menuBar()

    self.skeleton_widget = SkeletonWidget()
    self.skeleton_widget.resize(800, 600)

    self.setCentralWidget(self.skeleton_widget)

    self.implant_dock_widget = QDockWidget(self)
    self.implant_dock_widget.setWidget(ImplantWidget())
    self.coordinate_plot_dock_widget = QDockWidget(self)
    self.coordinate_plot_dock_widget.setWidget(CoordinatePlotWidget())

    self.view_menu = self.menu_bar.addMenu('View')
    view_implant_action = QAction('Implant', self)
    view_implant_action.triggered.connect(self.show_implant_stimulation)
    self.view_menu.addAction(view_implant_action)
    view_coordinates_action = QAction('Coordinates', self)
    view_coordinates_action.triggered.connect(self.show_coordinate_plots)
    self.view_menu.addAction(view_coordinates_action)

    wrist, shoulder = self.get_pos()

    wrist_pos_i = [0.2, round(0.4 - wrist[1], 1), round(-wrist[0], 1)]
    wrist_pos_f = [0.1, round(0.4 - shoulder[1], 1), round(-shoulder[0], 1)]

    angle_traj, time_traj = path_planning(wrist_pos_i, wrist_pos_f)

    angle_traj_2, time_traj_2 = path_planning(wrist_pos_f, wrist_pos_i)
    angle_traj = np.concatenate([angle_traj,angle_traj_2])
    time_traj_2 = time_traj_2 + np.max(time_traj)
    time_traj = np.concatenate([time_traj,time_traj_2])

    for i in range(10):
        angle_traj = np.concatenate([angle_traj,angle_traj])
        time_traj_add = time_traj + np.max(time_traj)
        time_traj = np.concatenate([time_traj,time_traj_add])

    self.set_trajectory(angle_traj[:,1], time_traj)

    self.timer = QtCore.QTimer(self)
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), lambda: self.update())
    update_interval = 100
    self.timer.start(update_interval)

    self.has_traj = False

    self.features_update = None
    self.results = None
    self.rgb_image = None

  @Slot()
  def update(self):
    self.update_model()
    self.update_view()

  def update_model(self):
    self.current_frame += 1

    #self.frame_indices = self.frame_indices[1:]
    #self.frame_indices.append(self.frame_indices[-1] + 1)
    self.frame_indices.append(self.current_frame)

    self.rgb_image, self.depth_image = self.image_data_provider.retrieve_rgb_depth_image()

    # Color image of the depth
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.05), cv2.COLORMAP_JET)

    cv2.imshow('RealSense', depth_colormap)

    # Estimate the pose with MediaPipe
    self.raw_joint_positions, self.filtered_joint_positions, self.raw_bones, self.filtered_bones, self.results = \
      estimate_pose(self.pose, self.rgb_image, self.depth_image, self.image_data_provider.depth_scale)

    mp_drawing.draw_landmarks(
      self.rgb_image,
      self.results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('MediaPipe Pose', self.rgb_image)

    x, y, z = self.raw_joint_positions[self.coordinate_plot_dock_widget.widget().selected_joint]
    filtered_x, filtered_y, filtered_z = self.filtered_joint_positions[self.coordinate_plot_dock_widget.widget().selected_joint]

    self.features_update = {'x_val': x, 'y_val': y, 'z_val': z, 'filtered_z_val': filtered_z}
  
  def update_view(self):
    self.coordinate_plot_dock_widget.widget().update(self.frame_indices, self.features_update)

    if self.has_traj:
      # Update angle_traj_widget by updating angle and time values:
      self.angle_traj_widget.update_plot(self.results.pose_world_landmarks.landmark, self.rgb_image)

    self.skeleton_widget.update_plot_data(self.raw_joint_positions, self.raw_bones, self.filtered_joint_positions, self.filtered_bones)

  @Slot()
  def show_implant_stimulation(self):
    self.implant_widget.show()

  @Slot()
  def show_coordinate_plots(self):
    self.coordinate_plot_widget.show()

  def get_pos(self):
    rgb_image, depth_image = self.image_data_provider.retrieve_rgb_depth_image()
    _, _, _, _, results = estimate_pose(self.pose, rgb_image, depth_image, self.image_data_provider.depth_scale)
    landmarks = results.pose_world_landmarks.landmark

    # Get coordinates
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    return wrist, shoulder

  def set_trajectory(self, angle_traj, time_traj):
    self.has_traj = True
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

      main_window = MainWindow(pose, image_data_provider)
      main_window.show()

      sys.exit(app.exec())
  finally:
    image_data_provider.stop()