from ximea import xiapi

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from .calibration import ImageCalibrator


class ImagePublisher(Node):
    """
    Create an ImagePublisher class, which is a subclass of the Node class.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_publisher')

        # Publishers for both raw and calibrated images
        self.publisher_raw = self.create_publisher(Image, '/ximea_frames_raw', 10)
        self.publisher_calibrated = self.create_publisher(Image, '/ximea_frames_calibrated', 10)
        
        # Publisher for camera info
        self.publisher_camera_info = self.create_publisher(CameraInfo, '/ximea_camera_info', 10)

        self.br = CvBridge()
        self.calibrator = ImageCalibrator()

        self.cam = xiapi.Camera()
        self.cam.open_device()
        # Settings
        self.cam.set_imgdataformat('XI_RGB24')
        self.cam.set_exposure(40000)
        self.cam.enable_auto_wb()
        self.cam.get_exposure()

        # Keep original image size
        self.cam.set_width(2064)
        self.cam.set_height(1544)

        # Create instance of Image to store image data and metadata
        self.img = xiapi.Image()
        # Start acquisition
        print('Starting data acquisition...')
        self.cam.start_acquisition()

        # Create CameraInfo message
        self.camera_info = CameraInfo()
        self.camera_info.width = 2064
        self.camera_info.height = 1544
        self.camera_info.distortion_model = 'plumb_bob'
        self.camera_info.d = [-0.378807, -0.215809, -0.000290376, -0.000209188, -0.0780538]  # Distortion coefficients
        self.camera_info.k = [1648.08815, 0.0, 1042.3339, 0.0, 1647.4375, 784.34486, 0.0, 0.0, 1.0]  # Camera matrix
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Rectification matrix
        # Adjust projection matrix (P) to reflect the ROI (width_pixels = 1769, height_pixels = 1225)
        self.camera_info.p = [1648.08815, 0.0, 1042.3339, 0.0, 0.0, 1647.4375, 784.34486, 0.0, 0.0, 0.0, 1.0, 0.0]
        
       
        self.camera_info.roi.x_offset = (2064 - 1769) // 2  # Center ROI in the original image width
       
        self.camera_info.roi.do_rectify = True  # Indicate that the image can be rectified using this ROI

    def publish_image(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        self.cam.get_image(self.img)
        data = self.img.get_image_data_numpy()

        if data is not None:
            # Publish raw image
            raw_image_message = self.br.cv2_to_imgmsg(data, encoding='bgr8')
            self.publisher_raw.publish(raw_image_message)

            # Publish calibrated image
            calibrated_data = self.calibrator.calibrate_for_numpy(data)
            calibrated_image_message = self.br.cv2_to_imgmsg(calibrated_data, encoding='bgr8')
            self.publisher_calibrated.publish(calibrated_image_message)

            # Publish camera info
            self.publisher_camera_info.publish(self.camera_info)

    def __del__(self):
        self.cam.stop_acquisition()
        self.cam.close_device()


def main(args=None):
    print("Driver Ximea started")
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_publisher = ImagePublisher()

    try:
        while True:
            image_publisher.publish_image()
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
