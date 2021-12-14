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

import matplotlib.pyplot as plt

from osim_env_main.opensim_environment import *
from osim_env_main.osim_model import *

## set PATH=C:\OpenSim 4.3\bin;%PATH% ##
dirname = os.path.dirname(QtCore.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

sys.path.insert(1, './')

from video_tracking.rt_pose_estimation import estimate_pose, bone_list, landmarks_list

import pyqtgraph.opengl as gl

import numpy as np

import mediapipe as mp

import csv

import cv2

import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
    print('Measurement')


def path_planning(wrist_pos_i, wrist_pos_f):
    traj = True  # to define wrist trajectory and write opensim trc file
    inv_kin = True  # to perform Inverse Kinematics with OpenSim

    if traj:  # define wrist trajectory with velocity bell shaped profile and write opensim trc file
        # movement definition
        movement = 'essaimediap5'
        # wrist_pos_i = [0.3, 0.7, 0.2]  # initial wrist position
        # wrist_pos_f = [0.4, 0.6, 0.2]  # final wrist position
        period = 10  # time to reach target in sec
        trajectory(movement, wrist_pos_i, wrist_pos_f, period, plot=False)

    if inv_kin:  # perform Inverse Kinematics with OpenSim
        # OpenSim model
        osim_model = 'models/full_arm_wrist.osim'
        # lock/unlock model coordinates
        # coords = ['elbow_flexion'] 'shoulder_rot'
        # coords = ['shoulder_elev', 'elv_angle', 'pro_sup','deviation', 'flexion', 'wrist_hand_r1', 'wrist_hand_r3'] # enlever elbow flex pour exo car cette coordonnee pas être blockée si true en dessous
        # osim_model = lock_Coord(osim_model, coords, 'true')
        # IK
        movement = 'essaimediap5'
        trc_file = 'trajectories/' + movement + '/' + movement + '.trc'
        output_file = 'trajectories/' + movement + '/IK_' + movement + '.mot'
        perform_ik(osim_model, trc_file, output_file)

        # plot IK results
        joints_to_plot = ['shoulder_elev', 'elbow_flexion', 'shoulder_rot', 'elv_angle']
        # joints_to_plot = ['elbow_flexion']
        angle_traj, time_traj = plot_ik_results(output_file, joints_to_plot)
        return angle_traj, time_traj


def trajectory(movement, wrist_pos_i, wrist_pos_f, period, freq=10, plot=True):
    """Write markers position .trc file.

    Parameters
    ----------
    movement: string
        to save .trc file in /movement folder
    wrist_pos_i: array
        initial [wrist_x, wrist_y, wrist_z]
    wrist_pos_f: array
        final [wrist_x, wrist_y, wrist_z]
    period: float
        time to reach target
    freq: int
        position points freqency
    output_file: str
        path to save joint position .mot file

    Returns
    --------
    trc file
    """

    if not os.path.isdir('trajectories/' + movement):
        os.makedirs('trajectories/' + movement)
    # bell shape velocity profile
    time = np.linspace(0, period, period * freq)
    vx = (wrist_pos_f[0] - wrist_pos_i[0]) * (-4 * 15 * np.power(time, 3) / (period ** 4) +
                                              5 * 6 * np.power(time, 4) / (period ** 5) +
                                              3 * 10 * np.power(time, 2) / (period ** 3))
    vy = (wrist_pos_f[1] - wrist_pos_i[1]) * (-4 * 15 * np.power(time, 3) / (period ** 4) +
                                              5 * 6 * np.power(time, 4) / (period ** 5) +
                                              3 * 10 * np.power(time, 2) / (period ** 3))
    vz = (wrist_pos_f[2] - wrist_pos_i[2]) * (-4 * 15 * np.power(time, 3) / (period ** 4) +
                                              5 * 6 * np.power(time, 4) / (period ** 5) +
                                              3 * 10 * np.power(time, 2) / (period ** 3))
    # position from velocity
    px = np.zeros(len(vx))
    py = np.zeros(len(vx))
    pz = np.zeros(len(vx))
    px[0] = wrist_pos_i[0]
    py[0] = wrist_pos_i[1]
    pz[0] = wrist_pos_i[2]
    for t in range(1, len(vx)):
        px[t] = np.trapz(vx[:t], dx=1 / freq) + px[0]
        py[t] = np.trapz(vy[:t], dx=1 / freq) + py[0]
        pz[t] = np.trapz(vz[:t], dx=1 / freq) + pz[0]

    # write opensim trc file to perform Inverse Kinematics
    if not os.path.isdir('trajectories/' + movement):
        os.mkdir('trajectories/' + movement)
    file = open('trajectories/' + movement + '/' + movement + '.trc', 'w')
    file.write('PathFileType	4	(X/Y/Z)	traj.trc\n')
    file.write(
        'DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames\n')
    file.write('30.00	30.00\t' + str(len(time)) + '\t1	mm	30.00	1\t' + str(len(time)) + '\n')
    file.write('Frame#\tTime\twrist\t\t\n')
    file.write('\t\tX1\tY1\tZ1\n')
    file.write('\n')
    for t in range(len(px)):
        file.write(str(t) + '\t' + str(time[t]) + '\t' + str(px[t] * 1000) + '\t' + str(py[t] * 1000) + '\t' + str(
            pz[t] * 1000) + '\n')

    if plot:
        # plot wrist trajectory
        plt.figure()
        plt.plot(px, label='x')
        plt.plot(py, label='y')
        plt.plot(pz, label='z')
        plt.legend()
        plt.savefig('trajectories/' + movement + '/wrist_traj.png')


def perform_ik(model_file, trc_file, output_file):
    """Perform Inverse Kinematics using OpenSim.

    Parameters
    ----------
    model_file: str
        OpenSim model file (.osim)
    trc_file: str
        path to markers position .trc file
    output_file: str
        path to save joint position .mot file

    Returns
    ---------
    IK joints .mot file
    """
    # model
    model = OsimModel(model_file, 0.01, 0.0001, body_ext_force=None, visualize=True, save_kin=True, )
    model.reset()

    # Marker Data
    markerData = opensim.MarkerData(trc_file)
    initial_time = markerData.getStartFrameTime()
    final_time = markerData.getLastFrameTime()

    # Set the IK tool
    ikTool = opensim.InverseKinematicsTool()
    ikTool.setModel(model.model)
    ikTool.setName('IK')
    ikTool.setMarkerDataFileName(trc_file)
    ikTool.setStartTime(initial_time)
    ikTool.setEndTime(final_time)
    ikTool.setOutputMotionFileName(output_file)

    ikTool.run()


def plot_ik_results(IK_file, joints=['shoulder_elev', 'elv_angle', 'shoulder_rot', 'elbow_flexion']):
    """Plot IK joints results.

    Parameters
    ----------
    IK_file: str
        path to IK .mot file
    joints: string list
        joints to plot
    """
    # read .mot file
    IK_data = open(IK_file, 'r')
    lines = IK_data.readlines()
    IK_joints = lines[10].split()[1:]
    joints_i = np.zeros(len(joints))
    for i in range(len(joints)):
        joints_i[i] = IK_joints.index(joints[i]) + 1
    joints_i = joints_i.astype(int)
    IK_angles = np.zeros((len(lines[11:]), len(joints)))
    IK_time = np.zeros(len(lines[11:]))
    for l in range(len(lines[11:])):
        IK_time[l] = lines[l + 11].split()[0]
        for i in range(len(joints)):
            IK_angles[l, i] = lines[l + 11].split()[joints_i[i]]

    # plot joints
    plt.figure()
    for i in range(len(joints)):
        plt.plot(IK_time, IK_angles[:, i], label=joints[i])
    plt.ylabel('Angle [°]')
    plt.legend()
    plt.title('IK joints')
    plt.xlabel('time [s]')
    plt.tight_layout()
    plt.savefig(os.path.dirname(IK_file) + '/IK_joints.png')
    return IK_angles, IK_time


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
        self.features_vals = {'x_val': [], 'filtered_x_val': [], 'y_val': [], 'filtered_y_val': [], 'z_val': [],
                              'filtered_z_val': []}

        self.graphWidget.setTitle("Real vs. filtered 3D coordinates")
        self.graphWidget.setLabel('left', "Coordinate value")
        self.graphWidget.setLabel('bottom', "Frame index")
        self.graphWidget.setBackground('w')
        self.graphWidget.addLegend()

        # Names of features currently displayed on plot
        self.visible_features = ['x_val', 'filtered_x_val']

        self.feature_pens = {'x_val': pen1, 'filtered_x_val': pen2, 'y_val': pen1, 'filtered_y_val': pen2,
                             'z_val': pen1, 'filtered_z_val': pen2}

        self.visible_line_refs = {}

        self.draw_visible_lines()

        self.layout = QtWidgets.QVBoxLayout(self)

        self.layout.addWidget(self.graphWidget)

        joint_choice_combo = QComboBox(self)
        for joint in landmarks_list:
            joint_choice_combo.addItem(str(joint))
        joint_choice_combo.currentIndexChanged.connect(
            lambda: self.select_joint(landmarks_list[joint_choice_combo.currentIndex()]))
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

    def update_frame(self, frame_indices, features_update):
        self.frame_indices = frame_indices

        for feature_name, feature_val in features_update.items():
            self.features_vals[feature_name].append(feature_val)

        for visible_feature_name in self.visible_features:
            self.visible_line_refs[visible_feature_name].setData(self.frame_indices,
                                                                 self.features_vals[visible_feature_name])

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
                                                                               self.features_vals[visible_feature_name],
                                                                               pen=self.feature_pens[
                                                                                   visible_feature_name],
                                                                               name=visible_feature_name)

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

class AngleTraj(QtWidgets.QMainWindow):

    def __init__(self, angle_traj, time_traj):
        super(AngleTraj, self).__init__()

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setTitle("Comparison of planned and real angle trajectory", size="10pt")
        self.graphWidget.setLabel("left", "Angle (°)")
        self.graphWidget.setLabel("bottom", "Time (s)")
        #Add legend
        self.graphWidget.addLegend()
        #Add grid
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setXRange(0, 20)

        #hour = [1,2,3,4,5,6,7,8,9,10]
        #temperature = [30,32,34,32,33,31,29,32,35,45]

        self.graphWidget.plot(time_traj, angle_traj,name="Trajectory planned")
        self.graphWidget.plot([1], [30], name="Real", symbol='o', symbolSize=10, symbolBrush=('g'))

        self.timer = QtCore.QTimer(self)
        self.connect(self.timer, QtCore.SIGNAL("timeout()"), lambda: self.update_plot())
        update_interval = 100
        self.timer.start(update_interval)

        self.angle = 0
        self.time_t = 0

    @Slot()
    def update_plot(self):
        #pen = pg.mkPen(color=(255, 0, 0), width=15, style=QtCore.Qt.DashLine)
        self.graphWidget.plot([self.time_t], [self.angle], symbol='o', symbolSize=10, symbolBrush=('g'))

    def update_values(self,time_t,angle):
        self.time_t = time_t
        self.angle = angle
        if time_t > 20:
            self.graphWidget.setXRange(int(time_t)-20,int(time_t))

    def set_trajectory(self,angle_traj, time_traj):
        self.angle_traj = angle_traj
        self.time_traj = time_traj


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

        self.skeletons = {'raw': Skeleton([1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]),
                          'filtered': Skeleton([0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0])}

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
        raw_joint_positions, filtered_joint_positions, raw_bones, filtered_bones, results = estimate_pose(self.pose,
                                                                                                          rgb_image,
                                                                                                          depth_image,
                                                                                                          self.depth_scale,
                                                                                                          self.current_frame)
        self.results = results
        self.skeletons['raw'].joint_positions, self.skeletons['raw'].bones = raw_joint_positions, raw_bones
        self.skeletons['filtered'].joint_positions, self.skeletons[
            'filtered'].bones = filtered_joint_positions, filtered_bones
        planned_joint_positions, planned_bones = None, None
        expected_joint_positions, expected_bones = None, None

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

        mp_drawing.draw_landmarks(
            rgb_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Render curl counter
        # Setup status box
        cv2.rectangle(rgb_image, (0,0), (225,78), (0,0,255), -1)
        cv2.line(rgb_image,pt1=(100,0), pt2=(100,78), color=(255,255,255), thickness=2)
        cv2.line(rgb_image,pt1=(225,0), pt2=(225,78), color=(255,255,255), thickness=2)
        cv2.line(rgb_image,pt1=(0,78), pt2=(225,78), color=(255,255,255), thickness=2)

        # Rep data
        cv2.putText(rgb_image, 'REPS', (5,31),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(rgb_image, str(self.counter),
                    (25,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(rgb_image, 'STAGE', (110,31),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(rgb_image, self.stage,
                    (110,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Pose', rgb_image)

        self.current_frame += 1

        # self.frame_indices = self.frame_indices[1:]
        # self.frame_indices.append(self.frame_indices[-1] + 1)
        self.frame_indices.append(self.current_frame)

        selected_joint = self.coordinate_plot_widget.selected_joint

        x, y, z = raw_joint_positions[selected_joint]
        filtered_x, filtered_y, filtered_z = filtered_joint_positions[selected_joint]

        features_update = {'x_val': x, 'y_val': y, 'z_val': z, 'filtered_z_val': filtered_z}
        self.coordinate_plot_widget.update_frame(self.frame_indices, features_update)

        ###

        skeletons_pos = []
        skeletons_color = []

        for skeleton in self.skeletons.values():
            pos = np.empty([len(skeleton.joint_positions), 3])
            color = np.array([skeleton.joint_color for _ in range(len(skeleton.joint_positions))])
            idx = 0
            for joint_name, joint_position in skeleton.joint_positions.items():
                # raw_pos[idx] = joint_position
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
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        return wrist, shoulder

    def set_trajectory(self,angle_traj, time_traj):
        self.angle_traj_widget = AngleTraj(angle_traj, time_traj)
        self.angle_traj_widget.setWindowTitle('Trajectory window')
        self.angle_traj_widget.time_begin = time.time()
        self.angle_traj_widget.resize(800, 400)
        self.angle_traj_widget.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default='webcam')
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


            # countdown(5)

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
        if not args.source == 'webcam':
            pipeline_1.stop()
