import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

import pyrealsense2 as rs

import pyqtgraph as pg

import argparse

import os

from video_tracking.rt_pose_estimation import estimate_pose

class MyWidget(QtWidgets.QWidget):
  def __init__(self, pipeline, depth_scale):
    super().__init__()

    self.pipeline = pipeline
    self.depth_scale = depth_scale

    self.graphWidget = pg.PlotWidget()

    self.frame_indices = list(range(100))
    self.x_val = [random.randint(0, 100) for _ in range(100)]
    self.y_val = [random.randint(0, 100) for _ in range(100)]
    self.z_val = [random.randint(0, 100) for _ in range(100)]

    self.graphWidget.setTitle("Real vs. filtered 3D coordinates")
    self.graphWidget.setLabel('left', "Coordinate value")
    self.graphWidget.setLabel('bottom', "Frame index")
    self.graphWidget.setBackground('w')
    self.graphWidget.addLegend()

    pen1 = pg.mkPen(color='r', width=2)
    pen2 = pg.mkPen(color='g', width=2)
    pen3 = pg.mkPen(color='b', width=2)

    x_line_ref = self.graphWidget.plot(self.frame_indices, self.x_val, name='X', pen=pen1)
    y_line_ref = self.graphWidget.plot(self.frame_indices, self.y_val, name='Y', pen=pen2)
    z_line_ref = self.graphWidget.plot(self.frame_indices, self.z_val, name='Z', pen=pen3)

    self.layout = QtWidgets.QVBoxLayout(self)
    self.layout.addWidget(self.graphWidget)

    self.timer = QtCore.QTimer(self)
    y_seqs = [self.x_val, self.y_val, self.z_val]
    refs = [x_line_ref, y_line_ref, z_line_ref]
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), lambda: self.update_plot_data(y_seqs, refs))
    self.timer.start(1000)

    self.current_frame = 1
  
  def update_plot_data(self, y_seqs, refs):
    # to be check: right x,y and z
    frames_1 = self.pipeline.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()
    color_frame_1 = frames_1.get_color_frame()

    estimate_pose(color_frame_1, depth_frame_1, self.depth_scale, self.current_frame)

    self.current_frame += 1

    self.frame_indices = self.frame_indices[1:]
    self.frame_indices.append(self.frame_indices[-1] + 1)

    for i, ref in enumerate(refs):
      y_seqs[i] = y_seqs[i][1:]
      y_seqs[i].append(random.randint(0, 100))

      ref.setData(self.frame_indices, y_seqs[i])

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
      print("Enabling device from file")
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

    app = QtWidgets.QApplication([])

    widget = MyWidget(pipeline_1, depth_scale)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
  finally:
    pipeline_1.stop()
    #np.savetxt('./output/Landmarks_coordinates_'+str(file_name)+'.csv',landmarks_coord,delimiter=',')