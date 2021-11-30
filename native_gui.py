from ntpath import join
import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

import pyrealsense2 as rs

import pyqtgraph as pg

import argparse

import os

from video_tracking.rt_pose_estimation import estimate_pose, bone_list

import pyqtgraph.opengl as gl

import numpy as np

import mediapipe as mp

mp_pose = mp.solutions.pose

class MyWidget(QtWidgets.QWidget):
  def __init__(self, pipeline, depth_scale, pose):
    super().__init__()

    self.pose = pose

    self.pipeline = pipeline
    self.depth_scale = depth_scale

    self.current_frame = 0

    self.graphWidget = pg.PlotWidget()

    self.frame_indices = list(range(self.current_frame))
    self.x_val = []
    self.y_val = []
    self.z_val = []

    self.graphWidget.setTitle("Real vs. filtered 3D coordinates")
    self.graphWidget.setLabel('left', "Coordinate value")
    self.graphWidget.setLabel('bottom', "Frame index")
    self.graphWidget.setBackground('w')
    self.graphWidget.addLegend()

    pen1 = pg.mkPen(color='r', width=2)
    pen2 = pg.mkPen(color='g', width=2)
    pen3 = pg.mkPen(color='b', width=2)

    self.x_line_ref = self.graphWidget.plot(self.frame_indices, self.x_val, name='X', pen=pen1)
    self.y_line_ref = self.graphWidget.plot(self.frame_indices, self.y_val, name='Y', pen=pen2)
    self.z_line_ref = self.graphWidget.plot(self.frame_indices, self.z_val, name='Z', pen=pen3)

    self.layout = QtWidgets.QVBoxLayout(self)
    #self.layout.addWidget(self.graphWidget)

    ###

    w = gl.GLViewWidget()

    self.pos = np.array([1, 1, 3])
    self.sp2 = gl.GLScatterPlotItem(pos=self.pos)

    g = gl.GLGridItem()
    w.addItem(g)

    # Bone drawing
    self.bone_item_positions = []
    self.bone_items = []
    for i in range(len(bone_list)):
      self.bone_item_positions.append(np.array([[0, 0, 0], [0, 0, 0]]))
      self.bone_items.append(gl.GLLinePlotItem(pos=self.bone_item_positions[i], width=1))
      w.addItem(self.bone_items[i])

    w.addItem(self.sp2)

    ###

    self.layout.addWidget(w)

    self.timer = QtCore.QTimer(self)
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), lambda: self.update_plot_data())
    update_interval = 100
    self.timer.start(update_interval)
  
  def update_plot_data(self):
    # to be check: right x,y and z
    frames_1 = self.pipeline.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()
    color_frame_1 = frames_1.get_color_frame()

    x, y, z, joint_positions, bones = estimate_pose(self.pose, color_frame_1, depth_frame_1, self.depth_scale, self.current_frame)

    self.current_frame += 1

    #self.frame_indices = self.frame_indices[1:]
    #self.frame_indices.append(self.frame_indices[-1] + 1)
    self.frame_indices.append(self.current_frame)

    self.x_val.append(x)
    self.y_val.append(y)
    self.z_val.append(z)

    self.x_line_ref.setData(self.frame_indices, self.x_val)
    self.y_line_ref.setData(self.frame_indices, self.y_val)
    self.z_line_ref.setData(self.frame_indices, self.z_val)

    ###

    pos = np.empty([len(joint_positions), 3])
    idx = 0
    for joint_name, joint_position in joint_positions.items():
      pos[idx, 0] = joint_position[0]
      pos[idx, 1] = joint_position[1]
      pos[idx, 2] = joint_position[2]
      idx += 1
    self.sp2.setData(pos=pos)

    for i, bone in enumerate(bones):
      x1, y1, z1, x2, y2, z2 = bone
      self.bone_item_positions[i] = np.array([[x1, y1, z1], [x2, y2, z2]])
      self.bone_items[i].setData(pos=self.bone_item_positions[i])

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