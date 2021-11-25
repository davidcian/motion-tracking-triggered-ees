import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


#landmarks_list = [mp_pose.PoseLandmark.LEFT_WRIST,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_SHOULDER,
#                  mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_SHOULDER]


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

########################################################################################################################


# Configure depth and color streams...
# ...from Camera 1

ctx = rs.context()
devices = ctx.query_devices()

pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(str(devices[0].get_info(rs.camera_info.serial_number)))
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #useful?

def filter_z(depth_values, current_z, median_z):
  max_z_deviation_up = 0.4
  max_z_deviation_down = 0.4
  #if current_z > depth_values[-1] + max_z_deviation_up or current_z < depth_values[-1] - max_z_deviation_down:
    #filtered_z = depth_values[-1]
  if current_z > median_z + max_z_deviation_up or current_z < median_z - max_z_deviation_down:
    filtered_z = depth_values[-1]
  else:
    filtered_z = current_z

  return filtered_z

#colorVideoCam1 = cv2.VideoWriter(save_dir+ filename[:-4]+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))

# Start streaming from camera
profile = pipeline_1.start(config_1)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

current_frame = 1

with mp_pose.Pose(static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    raw_z_values = []
    depth_values = []

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

        # if cv2.waitKey(25)==113: #q pressed
        #         break

        image = color_image_1

        image_height, image_width, _ = image.shape
        results = pose.process(image)

        if not results.pose_landmarks:
          continue


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

        depth_z = depth_scale * np.mean(depth_image_1[y-2:y+3,x-2:x+3])

        # Record the raw depth values
        raw_z_values.append(depth_z)

        window_len = 10

        median_filter_window = raw_z_values[max(0, len(raw_z_values) - window_len):len(raw_z_values)]
        # Apply a median filter to the depth value
        median_z = median_filter_window[len(median_filter_window) // 2]

        if current_frame > 1:
          filtered_z = filter_z(depth_values, depth_z, median_z)
        else:
          filtered_z = depth_z 

        depth_values.append(filtered_z)

       # Calculating the fps
      
       # time when we finish processing for this frame
        new_frame_time = time.time()
       # fps will be number of frame processed in given time frame
       # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
 
        # converting the fps into integer
        fps = int(fps)
 
        # putting the FPS as well as x, y and depth coordinates count on the frame
        cv2.putText(image,'x '+str(x)+'y'+str(y)+'depth {depth_z:.2f}'+'FPS'+str(fps),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
        #cv2.imshow('RealSense', depth_colormap_1)
        cv2.imshow('MediaPipe Pose', image)
        #plot_landmarks(
        #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw the depth value over time
        plt.title("Depth over time")
        plt.scatter(current_frame, depth_z, c='b')
        plt.scatter(current_frame, filtered_z, c='r')

        plt.pause(0.05)
        current_frame += 1

        if cv2.waitKey(5) & 0xFF == 27:
          break

      plt.show()

    # to do: stop process at the end of video
    finally:
        pipeline_1.stop()
        #np.savetxt('./output/Landmarks_coordinates_'+str(file_name)+'.csv',landmarks_coord,delimiter=',')
