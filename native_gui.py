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

    #self.pixmap = QtGui.QPixmap("graphical_assets/empty-implant-image.png")

  def paintEvent(self, event):
    painter = QtGui.QPainter(self)
    pixmap = QtGui.QPixmap("empty-implant-image.png")

    painter.drawPixmap(self.rect(), pixmap)

class MyWidget(QtWidgets.QWidget):
  def __init__(self, pipeline, depth_scale, pose):
    super().__init__()

    self.pose = pose

    self.pipeline = pipeline
    self.depth_scale = depth_scale

    self.coordinate_plot_widget = CoordinatePlotWidget()

    self.implant_widget = ImplantWidget()

    self.current_frame = 0

    self.frame_indices = list(range(self.current_frame))

    self.layout = QtWidgets.QVBoxLayout(self)

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

    # Raw and filtered data bone drawing
    self.raw_bone_item_positions = []
    self.raw_bone_items = []
    self.filtered_bone_item_positions = []
    self.filtered_bone_items = []
    for i in range(len(bone_list)):
      self.raw_bone_item_positions.append(np.array([[0, 0, 0], [0, 0, 0]]))
      self.raw_bone_items.append(gl.GLLinePlotItem(pos=self.raw_bone_item_positions[i], width=1))
      self.filtered_bone_item_positions.append(np.array([[0, 0, 0], [0, 0, 0]]))
      self.filtered_bone_items.append(gl.GLLinePlotItem(pos=self.filtered_bone_item_positions[i], width=1))
      w.addItem(self.raw_bone_items[i])
      w.addItem(self.filtered_bone_items[i])

    w.addItem(self.sp2)

    ###

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

  def update_plot_data(self):
    # to be check: right x,y and z
    frames_1 = self.pipeline.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()
    color_frame_1 = frames_1.get_color_frame()

    # Estimate the pose with MediaPipe
    raw_joint_positions, filtered_joint_positions, raw_bones, filtered_bones = estimate_pose(self.pose, color_frame_1, depth_frame_1, self.depth_scale, self.current_frame)
    planned_joint_positions, planned_bones = None, None
    expected_joint_positions, expected_bones = None, None

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

    # Plot raw (red) & filtered data (blue) joints
    raw_joint_color = [1.0, 0, 0, 1.0]
    filtered_joint_color = [0, 1.0, 0, 1.0]

    raw_pos = np.empty([len(raw_joint_positions), 3])
    raw_color = np.array([raw_joint_color for _ in range(len(raw_joint_positions))])
    idx = 0
    for joint_name, joint_position in raw_joint_positions.items():
      #raw_pos[idx] = joint_position
      raw_pos[idx, 0] = joint_position[0]
      raw_pos[idx, 1] = joint_position[2]
      raw_pos[idx, 2] = joint_position[1]
      idx += 1

    filtered_pos = np.empty([len(filtered_joint_positions), 3])
    filtered_color = np.array([filtered_joint_color for _ in range(len(filtered_joint_positions))])
    idx = 0
    for joint_name, joint_position in filtered_joint_positions.items():
      filtered_pos[idx] = joint_position
      idx += 1

    all_pos = np.vstack([raw_pos, filtered_pos])
    all_color = np.vstack([raw_color, filtered_color])

    self.sp2.setData(pos=all_pos, color=all_color)

    ###

    raw_bone_color = [1.0, 0, 0, 1.0]
    filtered_bone_color = [0, 1.0, 0, 1.0]

    # Plot raw (red) and filtered data (blue) data bones
    for i, bone in enumerate(raw_bones):
      x1, z1, y1, x2, z2, y2 = bone
      self.raw_bone_item_positions[i] = np.array([[x1, y1, z1], [x2, y2, z2]])
      self.raw_bone_items[i].setData(pos=self.raw_bone_item_positions[i], color=raw_bone_color)

    for i, bone in enumerate(filtered_bones):
      x1, y1, z1, x2, y2, z2 = bone
      self.filtered_bone_item_positions[i] = np.array([[x1, y1, z1], [x2, y2, z2]])
      self.filtered_bone_items[i].setData(pos=self.filtered_bone_item_positions[i], color=filtered_bone_color)

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

    if args.with_depth == 'true':
      config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming from camera
    profile = pipeline_1.start(config_1)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    with mp_pose.Pose(static_image_mode=False,
      model_complexity=2,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:

      app = QtWidgets.QApplication([])

      widget = MyWidget(pipeline_1, depth_scale, pose)
      widget.resize(800, 600)
      widget.show()

      sys.exit(app.exec())
  finally:
    pipeline_1.stop()
    #np.savetxt('./output/Landmarks_coordinates_'+str(file_name)+'.csv',landmarks_coord,delimiter=',')