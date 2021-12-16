from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QPushButton, QComboBox
from PySide6.QtGui import QVector3D
import pyqtgraph.opengl as gl

import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from video_tracking.rt_pose_estimation import estimate_pose, bone_list, landmarks_list

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

class SkeletonWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()

    self.layout = QtWidgets.QVBoxLayout(self)

    self.skeletons = {'raw': Skeleton([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]), 'filtered': Skeleton([0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0])}

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

    # Draw bones
    for skeleton in self.skeletons.values():
      for bone_item in skeleton.bone_items:
        w.addItem(bone_item)

    self.layout.addWidget(w)

  def update_plot_data(self, rgb_image, raw_joint_positions, raw_bones, filtered_joint_positions, filtered_bones):
    self.skeletons['raw'].joint_positions, self.skeletons['raw'].bones = raw_joint_positions, raw_bones
    self.skeletons['filtered'].joint_positions, self.skeletons['filtered'].bones = filtered_joint_positions, filtered_bones

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