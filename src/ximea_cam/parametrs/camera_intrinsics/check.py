import pickle

# Загрузка калибровочных данных
with open('cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)
with open('dist.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)
with open('calibration.pkl', 'rb') as f:
    calibration = pickle.load(f)

print(f'camera_matrix: {camera_matrix}')
print(f'dist_coeffs: {dist_coeffs}')
print(f'calibration: {calibration}')