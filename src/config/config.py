import numpy as np
from src.config.constants import G0

IMU_ERR = {
    # --- Гироскоп ---
    # Постоянное смещение нуля от запуска к запуску.
    # В реальности компенсируется алгоритмом при старте, поэтому задаем остаточную ошибку (около нуля).
    "gyro_b": np.array([0.5, 0.5, 0.5]),
    # "gyro_b": np.array([360.0, 360.0, 360.0]),
    "gyro_arw": np.array([0.09, 0.09, 0.09]),  # Случайное угловое блуждание (ARW)
    "gyro_b_stability": np.array([5, 5, 5]),  # Нестабильность смещения нуля в запуске
    "gyro_b_corr": np.array([300.0, 300.0, 300.0]),
    # --- Акселерометр ---
    # Постоянное смещение нуля. Также скомпенсировано на заводе. Оставляем микроскопическую остаточную ошибку.
    "accel_b": np.array([0.03, 0.03, 0.05]) * G0 * 1e-3,
    # "accel_b": np.array([1.0, 1.0, 1.5]) * G0 * 1e-3,
    "accel_vrw": np.array([8, 8, 8]) * 1e-3,
    "accel_b_stability": np.array([3, 3, 5]) * G0 * 1e-6,
    "accel_b_corr": np.array([500.0, 500.0, 500.0]),
}

GPS_ERR = {
    "stdp": np.array([0.05, 0.05, 0.10]),  # σ позиции [E, N, U] в метрах (RTK fix)
    "stdv": np.array([0.05, 0.05, 0.05]),  # σ скорости [m/s]
}

# Значения соответствуют RTK fix.
# Если хотите имитировать «обычный» одночастотный приёмник — позиция σ ≈ 1.5–3 м, скорость σ ≈ 0.05–0.1 м/с.

# IMU_ERR_0 = {
#     # --- Гироскоп ---
#     # gyro bias, deg/hr
#     "gyro_b": np.array([360.0, 360.0, 360.0]),
#     # gyro angle random walk, deg/rt-hr
#     "gyro_arw": np.array([0.09, 0.09, 0.09]),
#     # gyro bias instability, deg/hr
#     "gyro_b_stability": np.array([5, 5, 5]),
#     # gyro bias instability correlation, sec.
#     # set this to 'inf' to use a random walk model
#     # set this to a positive real number to use a first-order Gauss-Markkov model
#     # 'gyro_b_corr': 1,
#     # --- Акселерометр ---
#     # accelerometer bias, m/s^2
#     "accel_b": np.array([1.0, 1.0, 1.5]) * G * 1e-3,
#     # accelerometer velocity random walk, m/s/rt-hr
#     "accel_vrw": np.array([8, 8, 8]) * 1e-3,
#     # accelerometer bias instability, m/s^2
#     "accel_b_stability": np.array([3, 3, 5]) * G * 1e-6,
#     # accelerometer bias instability correlation, sec. Similar to gyro_b_corr
#     # 'accel_b_corr': 1,
#     # # magnetometer noise std, uT
#     # 'mag_std': np.array([0.2, 0.2, 0.2])
# }

"""
## default IMU, magnetometer and GPS error profiles.
# low accuracy, from AHRS380
#http://www.memsic.cn/userfiles/files/Datasheets/Inertial-System-Datasheets/AHRS380SA_Datasheet.pdf
gyro_low_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                     'b_drift': np.array([10.0, 10.0, 10.0]) * D2R/3600.0,
                     'b_corr':np.array([100.0, 100.0, 100.0]),
                     'arw': np.array([0.75, 0.75, 0.75]) * D2R/60.0}
accel_low_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                      'b_drift': np.array([2.0e-4, 2.0e-4, 2.0e-4]),
                      'b_corr': np.array([100.0, 100.0, 100.0]),
                      'vrw': np.array([0.05, 0.05, 0.05]) / 60.0}

# mid accuracy, partly from IMU381
gyro_mid_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                     'b_drift': np.array([3.5, 3.5, 3.5]) * D2R/3600.0,
                     'b_corr':np.array([100.0, 100.0, 100.0]),
                     'arw': np.array([0.25, 0.25, 0.25]) * D2R/60}
accel_mid_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                      'b_drift': np.array([5.0e-5, 5.0e-5, 5.0e-5]),
                      'b_corr': np.array([100.0, 100.0, 100.0]),
                      'vrw': np.array([0.03, 0.03, 0.03]) / 60}

# high accuracy, partly from HG9900, partly from
# http://www.dtic.mil/get-tr-doc/pdf?AD=ADA581016
gyro_high_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
                      'b_drift': np.array([0.1, 0.1, 0.1]) * D2R/3600.0,
                      'b_corr':np.array([100.0, 100.0, 100.0]),
                      'arw': np.array([2.0e-3, 2.0e-3, 2.0e-3]) * D2R/60}
accel_high_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
                       'b_drift': np.array([3.6e-6, 3.6e-6, 3.6e-6]),
                       'b_corr': np.array([100.0, 100.0, 100.0]),
                       'vrw': np.array([2.5e-5, 2.5e-5, 2.5e-5]) / 60}
"""
