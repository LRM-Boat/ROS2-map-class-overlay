import pickle
import numpy as np
import cv2
import os

from ament_index_python.packages import get_package_share_directory

class ImageCalibrator:
    def __init__(self, calibration_file_path: str = None):
        package_name = 'ximea_cam'
        package_share_directory = get_package_share_directory(package_name)
        parameters_calibration_dir = os.path.join(package_share_directory, 'parameters', 'camera_intrinsics')
        
        self.calibration_file_path = calibration_file_path if calibration_file_path is not None  else os.path.join(parameters_calibration_dir, 'calibration.pkl')
        
        # Загрузка калибровочных данных из файла
        self.set_calibration_file_path(self.calibration_file_path)

        self.map1 = None
        self.map2 = None


    def validate_and_get_absolute_path(self, file_path):
        # Проверка, является ли путь абсолютным
        if not os.path.isabs(file_path):
            # Преобразование пути в абсолютный
            file_path = os.path.abspath(file_path)
        
        # Проверка наличия файла по этому пути
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file does not exist: {file_path}")
        
        return file_path

    def load_calibration_data(self, file_path: str):
        with open(file_path, "rb") as f:
            self.cameraMatrix, self.distCoeffs = pickle.load(f)
        #print(f"Loaded camera matrix: \n{self.cameraMatrix}")
        #print(f"Loaded distortion coefficients: \n{self.distCoeffs}")

    def set_calibration_file_path(self, path: str):
        assert isinstance(path, str), "The path must be a string"
        
        abspath = self.validate_and_get_absolute_path(path)

        self.calibration_file_path = abspath

        self.load_calibration_data(self.calibration_file_path)


    def init_undistort_map(self, image_size):
        new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, image_size, 1)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.cameraMatrix, self.distCoeffs, None, new_camera_matrix, image_size, cv2.CV_32FC1)
        print(f"New camera matrix: \n{new_camera_matrix}")
        print(f"ROI: \n{self.roi}")

    def calibrate_for_numpy(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        h, w = image.shape[:2]

        if self.map1 is None or self.map2 is None:
            self.init_undistort_map((w, h))

        calibrated_image = cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # Применение ROI для обрезки черных краев
        x, y, w, h = self.roi
        calibrated_image = calibrated_image[y:y+h, x:x+w]
        print(f'calibrated_image.shape: {calibrated_image.shape}')
        return calibrated_image
