import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class TfTestNode(Node):
    def __init__(self):
        super().__init__('tf_test_node')
        


        self.lookup_rate = self.declare_parameter('lookup_rate', 0.05).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.lidar_frame = self.declare_parameter('lidar_frame', 'velodyne').value
       


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.check_and_publish_transform)  # Периодический вызов функции
        self.transform_publisher = self.create_publisher(TransformStamped, '/velodyne_to_map_transform', 10)
        self.get_logger().info('dublicate tf node started')

    def check_and_publish_transform(self):
        try:
            # Запрашиваем трансформацию между `map` и `velodyne`
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'map', self.lidar_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=self.lookup_rate)
            )

            # Логирование трансформации
            #self.get_logger().info(f"Transform from 'map' to 'velodyne':")
            #self.get_logger().info(f"  Translation - x: {transform.transform.translation.x}, "
             #                      f"y: {transform.transform.translation.y}, "
              #                     f"z: {transform.transform.translation.z}")
            #self.get_logger().info(f"  Rotation - x: {transform.transform.rotation.x}, "
             #                      f"y: {transform.transform.rotation.y}, "
              #                     f"z: {transform.transform.rotation.z}, "
               #                    f"w: {transform.transform.rotation.w}")

            # Публикация трансформации в топик
            transform.header.stamp = self.get_clock().now().to_msg()  # Обновление времени
            self.transform_publisher.publish(transform)
            #self.get_logger().info("Published TransformStamped to /velodyne_to_map_transform")

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get transform: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TfTestNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
