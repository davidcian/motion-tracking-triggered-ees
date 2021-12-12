from ntpath import join
import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QPushButton, QComboBox
from PySide6.QtCore import Slot

from PySide6.QtGui import QVector3D

import pyrealsense2 as rs

import pyqtgraph as pg

import argparse

import os

from video_tracking.rt_pose_estimation import estimate_pose, bone_list, landmarks_list

import pyqtgraph.opengl as gl

import numpy as np

import mediapipe as mp

import csv

import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pen1 = pg.mkPen(color='r', width=2)
pen2 = pg.mkPen(color='g', width=2)
pen3 = pg.mkPen(color='b', width=2)

class CoordinatePlotWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()

    self.selected_joint = mp_pose.PoseLandmark.LEFT_WRIST

    self.graphWidget = pg.PlotWidget()

    self.frame_indices = []
    # Features tracked by the coordinate plot (e.g. Cartesian coordinates of a joint)
    self.features_vals = {'x_val': [], 'filtered_x_val': [], 'y_val': [], 'filtered_y_val': [], 'z_val': [], 'filtered_z_val': []}

    self.graphWidget.setTitle("Real vs. filtered 3D coordinates")
    self.graphWidget.setLabel('left', "Coordinate value")
    self.graphWidget.setLabel('bottom', "Frame index")
    self.graphWidget.setBackground('w')
    self.graphWidget.addLegend()

    # Names of features currently displayed on plot
    self.visible_features = ['x_val', 'filtered_x_val']

    self.feature_pens = {'x_val': pen1, 'filtered_x_val': pen2, 'y_val': pen1, 'filtered_y_val': pen2, 'z_val': pen1, 'filtered_z_val': pen2}

    self.visible_line_refs = {}

    self.draw_visible_lines()

    self.layout = QtWidgets.QVBoxLayout(self)
    
    self.layout.addWidget(self.graphWidget)

    joint_choice_combo = QComboBox(self)
    for joint in landmarks_list:
      joint_choice_combo.addItem(str(joint))
    joint_choice_combo.currentIndexChanged.connect(lambda: self.select_joint(landmarks_list[joint_choice_combo.currentIndex()]))
    self.layout.addWidget(joint_choice_combo)

    self.show_x_button = QPushButton("Show X coordinate")
    self.show_y_button = QPushButton("Show Y coordinate")
    self.show_z_button = QPushButton("Show Z coordinate")

    self.show_x_button.clicked.connect(self.show_x_coordinate_plot)
    self.show_y_button.clicked.connect(self.show_y_coordinate_plot)
    self.show_z_button.clicked.connect(self.show_z_coordinate_plot)

    self.layout.addWidget(self.show_x_button)
    self.layout.addWidget(self.show_y_button)
    self.layout.addWidget(self.show_z_button)

  @Slot()
  def select_joint(self, joint):
    self.selected_joint = joint

  def update(self, frame_indices, features_update):
    self.frame_indices = frame_indices

    for feature_name, feature_val in features_update.items():
      self.features_vals[feature_name].append(feature_val)

    for visible_feature_name in self.visible_features:
      self.visible_line_refs[visible_feature_name].setData(self.frame_indices, self.features_vals[visible_feature_name])

  @Slot()
  def show_x_coordinate_plot(self):
    self.clear_visible_lines()
    self.visible_features = ['x_val', 'filtered_x_val']
    self.draw_visible_lines()

  @Slot()
  def show_y_coordinate_plot(self):
    self.clear_visible_lines()
    self.visible_features = ['y_val', 'filtered_y_val']
    self.draw_visible_lines()

  @Slot()
  def show_z_coordinate_plot(self):
    self.clear_visible_lines()
    self.visible_features = ['z_val', 'filtered_z_val']
    self.draw_visible_lines()

  def clear_visible_lines(self):
    for visible_feature_name in self.visible_features:
      self.graphWidget.removeItem(self.visible_line_refs[visible_feature_name])

  def draw_visible_lines(self):
    for visible_feature_name in self.visible_features:
      if visible_feature_name not in self.visible_line_refs:
        self.visible_line_refs[visible_feature_name] = pg.PlotDataItem(self.frame_indices, 
          self.features_vals[visible_feature_name], pen=self.feature_pens[visible_feature_name], name=visible_feature_name)

      self.graphWidget.addItem(self.visible_line_refs[visible_feature_name])

class ImplantWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()
    
    self.scene = QtWidgets.QGraphicsScene(self)

    self.pixmap_res = QtGui.QPixmap("empty-implant-image.png")

    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.pixmap_res)

    self.scene.addItem(self.pixmap)

    self.view = QtWidgets.QGraphicsView(self.scene)

    self.view.show()

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
  def __init__(self, pose, pipeline=None, depth_scale=None, webcam=None):
    super().__init__()

    self.pose = pose

    self.pipeline = pipeline
    self.depth_scale = depth_scale

    self.coordinate_plot_widget = CoordinatePlotWidget()

    self.implant_widget = ImplantWidget()

    self.current_frame = 0

    self.frame_indices = list(range(self.current_frame))

    self.layout = QtWidgets.QVBoxLayout(self)

    self.skeletons = {'raw': Skeleton([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]), 'filtered': Skeleton([0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0])}

    self.webcam = webcam

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

  @Slot()
  def show_implant_stimulation(self):
    self.implant_widget.show()

  @Slot()
  def show_coordinate_plots(self):
    self.coordinate_plot_widget.show()

  def retrieve_data(self):
    if not self.webcam:
      frames = self.pipeline.wait_for_frames()
      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()

      depth_image = np.asanyarray(depth_frame.get_data())
      rgb_image = np.asanyarray(color_frame.get_data())
    else:
      ret_val, rgb_image = self.webcam.read()
      depth_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))

    return rgb_image, depth_image

  def update_plot_data(self):
    rgb_image, depth_image = self.retrieve_data()

    if not self.webcam:
      # Color image of the depth
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

      cv2.imshow('RealSense', depth_colormap)

    # Estimate the pose with MediaPipe
    raw_joint_positions, filtered_joint_positions, raw_bones, filtered_bones, results = estimate_pose(self.pose, rgb_image, depth_image, self.depth_scale, self.current_frame)
    self.skeletons['raw'].joint_positions, self.skeletons['raw'].bones = raw_joint_positions, raw_bones
    self.skeletons['filtered'].joint_positions, self.skeletons['filtered'].bones = filtered_joint_positions, filtered_bones
    planned_joint_positions, planned_bones = None, None
    expected_joint_positions, expected_bones = None, None

    mp_drawing.draw_landmarks(
      rgb_image,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--source", default='video')
  parser.add_argument("-d", "--dir", default="C:\\Users\\cleme\\Documents\\EPFL\\Master\\MA-3\\sensor\\data\\")
  parser.add_argument("-f", "--file", default='cam1_911222060374_record_30_09_2021_1404_05.avi')
  parser.add_argument("-z", "--depth-file")
  parser.add_argument("-w", "--with-depth")
  args = parser.parse_args()

  try:
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()

    if args.source == 'video':
      rs.config.enable_device_from_file(config_1, os.path.join(args.dir, args.depth_file))
      config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    elif args.source == 'live':
      ctx = rs.context()
      devices = ctx.query_devices()
      config_1.enable_device(str(devices[0].get_info(rs.camera_info.serial_number)))
      config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    elif args.source == 'webcam':
      cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if args.with_depth == 'true':
      config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if not args.source == 'webcam':
      # Start streaming from camera
      profile = pipeline_1.start(config_1)
      depth_sensor = profile.get_device().first_depth_sensor()
      depth_scale = depth_sensor.get_depth_scale()

    with mp_pose.Pose(static_image_mode=False,
      model_complexity=2,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:

      app = QtWidgets.QApplication([])

      if not args.source == 'webcam':
        widget = MyWidget(pose, pipeline=pipeline_1, depth_scale=depth_scale)
      else:
        widget = MyWidget(pose, webcam=cam)
      widget.resize(800, 600)
      widget.show()

      sys.exit(app.exec())
  finally:
    if not args.source == 'webcam':
      pipeline_1.stop()