import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from ament_index_python.packages import get_package_share_directory
import os
from sensor_msgs_py import point_cloud2 as pc2


class ONNXInferenceNode(Node):
    def __init__(self):
        super().__init__('onnx_inference_node')


        #переменные из launch фаила
        self.image_topic = self.declare_parameter('image_topic', "/image_raw").value
        self.lidar_topic = self.declare_parameter('lidar_topic', "/synced/velodyne_points").value
        self.transform_topic = self.declare_parameter('transform_topic', "/velodyne_to_map_transform").value



        # Подписки на топики
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10)
        
        self.transform_subscription = self.create_subscription(
            TransformStamped,
            self.transform_topic,
            self.transform_callback,
            10)
        
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10)

        # Публикации
        self.segmented_image_publisher = self.create_publisher(Image, '/segmented_image', 10)
        self.transform_publisher = self.create_publisher(TransformStamped, '/velodyne_saved_transform', 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/saved_lidar_points', 10)

        # ONNX модель
        package_share_directory = get_package_share_directory('onnx_segmentation')
        self.backend_files = [os.path.join(package_share_directory, 'data', 'deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.onnx')]
        self.model_cfg = os.path.join(package_share_directory, 'data/mmsegmentation/configs/deeplabv3plus', 'deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.py')
        self.deploy_cfg = os.path.join(package_share_directory, 'data/mmdeploy/configs/mmseg/', 'segmentation_onnxruntime_dynamic.py')

        self.bridge = CvBridge()
        self.device = 'cpu'

        self.color_map = np.array([
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ])

        deploy_cfg, model_cfg = load_config(self.deploy_cfg, self.model_cfg)
        self.get_logger().info('Model found')
        self.task_processor = build_task_processor(model_cfg, deploy_cfg, self.device)
        self.model = self.task_processor.build_backend_model(self.backend_files)
        self.input_shape = get_input_shape(deploy_cfg)
        self.get_logger().info('ONNX Inference Node Initialized')

        # Сохранение данных
        self.current_pointcloud = None
        self.buffer_pointcloud = None
        self.current_transform = None
        self.buffer_transform = None

    def lidar_callback(self, msg):
        self.current_pointcloud = msg
        

    def transform_callback(self, msg):
        self.current_transform = msg
        

    def image_callback(self, msg):
        #if not self.current_pointcloud or not self.current_transform:
         #   self.get_logger().warn("Waiting for both point cloud and transform data.")
          #  return

        # Проверка временных меток
        #image_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        #lidar_timestamp = self.current_pointcloud.header.stamp.sec + self.current_pointcloud.header.stamp.nanosec * 1e-9
        #transform_timestamp = self.current_transform.header.stamp.sec + self.current_transform.header.stamp.nanosec * 1e-9

        #max_time_diff = 0.5
        #if abs(image_timestamp - lidar_timestamp) > max_time_diff or abs(image_timestamp - transform_timestamp) > max_time_diff:
           # self.get_logger().error(f"Timestamp mismatch: Image({image_timestamp}), Lidar({lidar_timestamp}), Transform({transform_timestamp})")
           # return

        try:
            #self.buffer_transform = self.current_transform
            #self.buffer_pointcloud = self.current_pointcloud

            # Конвертируем ROS Image в OpenCV формат
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            resized_image = cv2.resize(cv_image, (1024, 720))

            # Выполняем инференс
            model_inputs, _ = self.task_processor.create_input(resized_image, self.input_shape)
            result = self.model.test_step(model_inputs)
            pred_mask = result[0].pred_sem_seg.data.cpu().numpy().squeeze()

            # Генерация цветного изображения
            segmented_image = self.create_colored_image_from_classes(pred_mask)
            resized_image = cv2.resize(segmented_image, (2064, 1544))

            # Публикация сегментированного изображения
            segmented_image_msg = self.bridge.cv2_to_imgmsg(resized_image, encoding="bgr8")
            segmented_image_msg.header = msg.header
            self.segmented_image_publisher.publish(segmented_image_msg)

            # Публикация лидара и трансформации
            #self.pointcloud_publisher.publish(self.buffer_pointcloud)
            #self.transform_publisher.publish(self.buffer_transform)
            self.get_logger().info("Data was send")

        except Exception as e:
            self.get_logger().error(f"Error during ONNX inference: {e}")

    def create_colored_image_from_classes(self, class_mask):
        height, width = class_mask.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color in enumerate(self.color_map):
            color_image[class_mask == class_id] = color

        return color_image


def main(args=None):
    rclpy.init(args=args)
    node = ONNXInferenceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
