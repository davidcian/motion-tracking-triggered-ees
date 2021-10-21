import cv2
import mediapipe as mp
import av
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

from help_viz import plot_landmarks

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



########################################################################################################################

load_dir = "C:\\Users\\cleme\\Documents\\EPFL\\Master\\MA-3\\sensor\\data\\"
file_name = 'cam1_911222060374_record_30_09_2021_1404_05'

########################################################################################################################

container = av.open(str(load_dir)+str(file_name)+".avi")

# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
rs.config.enable_device_from_file(config_1, load_dir + file_name +".bag")
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #useful?



#colorVideoCam1 = cv2.VideoWriter(save_dir+ filename[:-4]+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))

# Start streaming from camera
profile = pipeline_1.start(config_1)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

frame_id = 0


with mp_pose.Pose(static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  #for frame in container.decode(video=0):
    try:
      while True:

        # to be check: right x,y and z
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
        for landmark in landmarks_list:
            coord = results.pose_landmarks.landmark[landmark]
            x = min(int(coord.x * image_width), 640-1)
            y = min(int(coord.y * image_height), 480-1)
            depth_z = depth_scale * depth_image_1[y,x]
            # if depth_z == 0:
            #     print(depth_image_1.shape)
            #     print(depth_image_1)
            landmarks_coord.append([frame_id,landmark,coord.x * image_width,coord.y * image_height,coord.z,depth_z])
        #print(color_image_1)

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
        y = x = min(int(coord.y * image_height), 480-1)
        depth_z = depth_scale * depth_image_1[y,x]

        cv2.putText(image,'x'+str(x)+'y'+str(y)+'depth'+str(depth_z),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.imshow('RealSense', depth_colormap_1)
        cv2.imshow('MediaPipe Pose', image)
        #plot_landmarks(
        #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        if cv2.waitKey(5) & 0xFF == 27:
          break
        frame_id = frame_id + 1

    # to do: stop process at the end of video
    finally:
        pipeline_1.stop()
        np.savetxt('./output/Landmarks_coordinates_'+str(file_name)+'.csv',landmarks_coord,delimiter=',')
