import numpy as np
import matplotlib.pyplot as plt
import os
from opensim_environment import *
from osim_model import *

def main():

    traj = True  # to define wrist trajectory and write opensim trc file
    inv_kin = True  # to perform Inverse Kinematics with OpenSim

    if traj:   # define wrist trajectory with velocity bell shaped profile and write opensim trc file
        # movement definition
        movement = 'example'
        wrist_pos_i = [0.3, 0.7, 0.2]  # initial wrist position
        wrist_pos_f = [0.4, 0.6, 0.2]  # final wrist position
        period = 1  # time to reach target in sec
        trajectory(movement, wrist_pos_i, wrist_pos_f, period, plot=True)

    if inv_kin:  # perform Inverse Kinematics with OpenSim
        # OpenSim model
        osim_model = 'models/full_arm_wrist.osim'
        # lock/unlock model coordinates
        coords = ['shoulder_elev', 'elbow_flexion', 'shoulder_rot', 'elv_angle']
        osim_model = lock_Coord(osim_model, coords, 'false')
        # IK
        movement = 'example'
        trc_file = 'trajectories/'+movement+'/'+movement+'.trc'
        output_file = 'trajectories/'+movement+'/IK_'+movement+'.mot'
        perform_ik(osim_model, trc_file, output_file)

        # plot IK results
        joints_to_plot = ['shoulder_elev', 'elbow_flexion', 'shoulder_rot', 'elv_angle']
        plot_ik_results(output_file, joints_to_plot)

    plt.show()


def trajectory(movement, wrist_pos_i, wrist_pos_f, period, freq=50, plot=True):
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

    if not os.path.isdir('trajectories/'+movement):
        os.makedirs('trajectories/'+movement)
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
    model = OsimModel(model_file, 0.01, 0.0001, body_ext_force=None, visualize=True, save_kin=True,)
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
    plt.ylabel('Angle [??]')
    plt.legend()
    plt.title('IK joints')
    plt.xlabel('time [s]')
    plt.tight_layout()
    plt.savefig(os.path.dirname(IK_file)+ '/IK_joints.png')


if __name__ == '__main__':
        main()
