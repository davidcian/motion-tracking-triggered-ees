import cv2
import numpy as np
import pyrealsense2 as rs

class WebcamProvider():
  def __init__(self):
    self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    self.width  = int(self.webcam.get(cv2.CV_CAP_PROP_FRAME_WIDTH))
    self.height = int(self.webcam.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))

    self.depth_scale = 1

  def retrieve_rgb_image(self):
    ret_val, rgb_image = self.webcam.read()
    return rgb_image

  def retrieve_depth_image(self):
    depth_image = np.zeros((self.width, self.height))
    return depth_image

  def retrieve_rgb_depth_image(self):
    rgb_image = self.retrieve_rgb_image()
    depth_image = self.retrieve_depth_image()

    return rgb_image, depth_image

  def stop(self):
    pass

class PyRealSenseProvider():
  def __init__(self):
    self.pipeline = rs.pipeline()
    self.config = rs.config()

  def start(self):
    self.profile = self.pipeline.start(self.config)
    self.depth_sensor = self.profile.get_device().first_depth_sensor()
    self.depth_scale = self.depth_sensor.get_depth_scale()

  def retrieve_rgb_depth_image(self):
    frames = self.pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    rgb_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return rgb_image, depth_image

  def stop(self):
    self.pipeline.stop()

class PyRealSenseCameraProvider(PyRealSenseProvider):
  def __init__(self, with_depth=True):
    super().__init__()

    ctx = rs.context()
    devices = ctx.query_devices()
    self.config.enable_device(str(devices[0].get_info(rs.camera_info.serial_number)))
    self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    if with_depth:
      self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    super().start()

class PyRealSenseVideoProvider(PyRealSenseProvider):
  def __init__(self, file_path, with_depth=True):
    super().__init__()

    rs.config.enable_device_from_file(self.config, file_path)
    self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    if with_depth:
      self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    super().start()