import numpy as np
import time
import os
import matplotlib.pyplot as plt

from osim_env_main.opensim_environment import *
from osim_env_main.osim_model import *

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Slot
import pyqtgraph as pg

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
    osim_model = 'D:\motion-tracking-triggered-ees\osim_env_main\models\\full_arm_wrist.osim'
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
  file.write('DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames\n')
  file.write('30.00	30.00\t' + str(len(time)) + '\t1	mm	30.00	1\t' + str(len(time)) + '\n')
  file.write('Frame#\tTime\twrist\t\t\n')
  file.write('\t\tX1\tY1\tZ1\n')
  file.write('\n')
  for t in range(len(px)):
    file.write(str(t) + '\t' + str(time[t]) + '\t' + str(px[t] * 1000) + '\t' + str(py[t] * 1000) + '\t' + str(pz[t] * 1000) + '\n')

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