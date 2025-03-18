from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'onnx_segmentation'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

# Все файлы из папки data/mmdeploy
mmdeploy_files = package_files('data/mmdeploy')
# Все файлы из папки data/mmsegmentation
mmsegmentation_files = package_files('data/mmsegmentation')


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         # Скопировать все файлы и папки из папки data
        (os.path.join('share', package_name, 'data'), ['data/end2end.onnx', 'data/demo.png', 'data/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.onnx']),
        # base mmdeploy
        (os.path.join('share', package_name, 'data/mmdeploy/configs/_base_/backends/'), glob('data/mmdeploy/configs/_base_/backends/*')),
        (os.path.join('share', package_name, 'data/mmdeploy/configs/_base_'), ['data/mmdeploy/configs/_base_/onnx_config.py', 'data/mmdeploy/configs/_base_/torchscript_config.py']),
        
        #mmseg mmdeploy
        (os.path.join('share', package_name, 'data/mmdeploy/configs/mmseg/'), glob('data/mmdeploy/configs/mmseg/*')),
        #mmdeploy mmdeploy
        (os.path.join('share', package_name, 'data'), ['data/mmdeploy/mmdeploy/__init__.py', 'data/mmdeploy/mmdeploy/version.py']),
        #mmdeploy backend
        (os.path.join('share', package_name, 'data'), ['data/mmdeploy/mmdeploy/backend/__init__.py']),
        (os.path.join('share', package_name, 'data'), glob('data/mmdeploy/mmdeploy/backend/onnxruntime/*')),

   
        
        #mmsegmentationonnx_segmentationation/configs/_base_/schedules/'), glob('data/mmsegmentation/configs/_base_/schedules/*')),
        
        (os.path.join('share', package_name, 'data/mmsegmentation/configs/_base_/models'), glob('data/mmsegmentation/configs/_base_/models/*')),
        #deeplabv3plus
        (os.path.join('share', package_name, 'data/mmsegmentation/configs/deeplabv3plus/'), glob('data/mmsegmentation/configs/deeplabv3plus/*.py')),
        (os.path.join('share', package_name, 'data/mmsegmentation/configs/_base_/datasets/'), glob('data/mmsegmentation/configs/_base_/datasets/*')),
        #deeplabv3
        (os.path.join('share', package_name, 'data/mmsegmentation/configs/deeplabv3/'), glob('data/mmsegmentation/configs/deeplabv3/*.py')),
        (os.path.join('share', package_name, 'data/mmsegmentation/configs/_base_'), ['data/mmsegmentation/configs/_base_/default_runtime.py']),

        (os.path.join('share', package_name, 'data/mmsegmentation/configs/_base_/schedules'), glob('data/mmsegmentation/configs/_base_/schedules/*.py'))
         
        
        
        
        
        
    ],
    install_requires=['setuptools', 'rclpy', 'cv_bridge', 'mmdeploy', 'torch', 'opencv-python'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='anton.belolipetskij@gmail.com',
    description='Package for ONNX segmentation inference',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'onnx_segmentation_node = onnx_segmentation.onnx_segmentation_node:main',
            'tests= onnx_segmentation.tests:main',
        ],
    },
)

