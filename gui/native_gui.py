from ntpath import join
import sys
import random
import argparse
import os
import csv

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QPushButton, QComboBox, QMainWindow
from PySide6.QtCore import Slot
import pyqtgraph as pg

import numpy as np
import mediapipe as mp

from image_data_providers import PyRealSenseCameraProvider, PyRealSenseVideoProvider, WebcamProvider

from opensim_tools import calculate_angle, AngleTraj, path_planning

from skeleton_display import SkeletonWidget

mp_pose = mp.solutions.pose

pen1 = pg.mkPen(color='r', width=2)
pen2 = pg.mkPen(color='g', width=2)
pen3 = pg.mkPen(color='b', width=2)

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

      widget = SkeletonWidget(pose, image_data_provider)
      widget.resize(800, 600)

      main_window = QMainWindow()
      main_window.setCentralWidget(widget)
      main_window.show()

      wrist, shoulder = widget.get_pos()

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

      widget.set_trajectory(angle_traj[:,1], time_traj)

      sys.exit(app.exec())
  finally:
    image_data_provider.stop()