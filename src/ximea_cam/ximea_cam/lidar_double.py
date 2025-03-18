import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class VelodynePointsDuplicator(Node):
    def __init__(self):
        super().__init__('velodyne_points_duplicator')
        
        # Подписка на исходный топик /velodyne_points
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.callback,
            10)
        
        # Публикация в дублирующий топик /velodyne_points_duble
        self.publisher = self.create_publisher(PointCloud2, '/velodyne_points_duble', 10)
        
        self.get_logger().info("VelodynePointsDuplicator node has been started.")

    def callback(self, msg):
        # Публикация сообщения в топик /velodyne_points_duble
        self.publisher.publish(msg)
        self.get_logger().info("Message duplicated to /velodyne_points_duble")

def main(args=None):
    rclpy.init(args=args)
    node = VelodynePointsDuplicator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
