from setuptools import setup
from glob import glob
import os

package_name = 'ximea_cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'parameters', 'camera_intrinsics'), glob('parametrs/camera_intrinsics/*.pkl')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kulikov',
    maintainer_email='nikita.kulikov.13@yandex.ru',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'ximea_frame_publisher = ximea_cam.driver_ximea:main',
            'calibration = ximea_cam.calibration',
            'lidar_duble = ximea_cam.lidar_double:main',
        ],
    },
)
