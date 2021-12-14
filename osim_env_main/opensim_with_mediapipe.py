import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import os
from opensim_environment import *
from osim_model import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

landmarks_coord = []

landmarks_list = [mp_pose.PoseLandmark.LEFT_WRIST,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_SHOULDER]


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0
depth_z_1 = 0

ctx = rs.context()
devices = ctx.query_devices()

pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(str(devices[0].get_info(rs.camera_info.serial_number)))
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #useful?

# Start streaming from camera
profile = pipeline_1.start(config_1)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

frame_id = 0

def main():

    traj = True  # to define wrist trajectory and write opensim trc file
    inv_kin = True  # to perform Inverse Kinematics with OpenSim
    Xtarget = 0.4
    Ytarget = 0.6
    Depth_Target = 0.2
    if traj:   # define wrist trajectory with velocity bell shaped profile and write opensim trc file
        # movement definition
        movement = 'try3'
        wrist_pos_i = [x/640, y/480, depth_z/depth_scale]  # initial wrist position
        wrist_pos_f = [Xtarget, Ytarget, Depth_Target]  # final wrist position
        period = int((np.sqrt((x/640-Xtarget)**2+(y/480-Ytarget)**2)+(depth_z-Depth_Target)**2)/0.10)  # time to reach target in sec
        trajectory(movement, wrist_pos_i, wrist_pos_f, period, plot=True)

    if inv_kin:  # perform Inverse Kinematics with OpenSim
        # OpenSim model
        osim_model = 'models/full_arm_wrist.osim'
        # lock/unlock model coordinates
        coords = ['shoulder_elev', 'elbow_flexion', 'shoulder_rot', 'elv_angle']
        osim_model = lock_Coord(osim_model, coords, 'false')
        # IK
        movement = 'try3'
        trc_file = 'trajectories/'+movement+'/'+movement+'.trc'
        output_file = 'trajectories/'+movement+'/IK_'+movement+'.mot'
        perform_ik(osim_model, trc_file, output_file)

        # plot IK results
        joints_to_plot = ['shoulder_elev', 'elbow_flexion', 'shoulder_rot', 'elv_angle']
        #plot_ik_results(output_file, joints_to_plot)

    #plt.show()

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
    plt.ylabel('Angle [°]')
    plt.legend()
    plt.title('IK joints')
    plt.xlabel('time [s]')
    plt.tight_layout()
    plt.savefig(os.path.dirname(IK_file)+ '/IK_joints.png')




with mp_pose.Pose(static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    raw_z_values = []
    main()

    try:
      while True:
        
        # to be check: right x,y and z
        start = time.time()
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())


        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.05), cv2.COLORMAP_JET)

        if cv2.waitKey(25)==113: #q pressed
                break


        #frame = frame.reformat(frame.width, frame.height, 'rgb24')
        #image = frame.to_ndarray()

        # Convert the BGR image to RGB before processing.
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(color_image_1,cv2.COLOR_BGR2RGB)
        image = color_image_1

        image_height, image_width, _ = image.shape
        results = pose.process(image)

        if not results.pose_landmarks:
          continue
        

        # print(
        #     f'Left wrist coordinates: ('
        #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width}, '
        #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height})'
        # )

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        coord = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        x = min(int(coord.x * image_width), 640-1)
        y = min(int(coord.y * image_height), 480-1)
        matrix_depth = depth_image_1[y-1:y+2,x-1:x+2]
        depth_z = depth_scale * np.mean(matrix_depth)
        
            
        if (x != px or y != py or depth_z != Depth_Target)
        
       
        #Calculating the fps
        
        end = time.time()
        totalTime = end - start
        fps = 1/totalTime
 
        # putting the FPS as well as x, y and depth coordinates count on the frame
        cv2.putText(image,'x'+str(x)+'y'+str(y)+'depth {0:.4f}'.format(depth_z)+'FPS'+str(fps),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
        #cv2.imshow('RealSense', depth_colormap_1)
        cv2.imshow('MediaPipe Pose', image)
        #plot_landmarks(
        #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        if cv2.waitKey(5) & 0xFF == 27:
          break
        frame_id = frame_id + 1
        
    # to do: stop process at the end of video
    finally:
        pipeline_1.stop()
        