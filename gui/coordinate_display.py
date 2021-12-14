from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QPushButton, QComboBox
import pyqtgraph as pg

import mediapipe as mp

from video_tracking.rt_pose_estimation import landmarks_list

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