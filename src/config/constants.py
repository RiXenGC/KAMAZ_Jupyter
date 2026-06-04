import numpy as np

# Constants for project
# Geodesy (WGS-84)
A_WGS84 = 6378137.0  # большая полуось, м
E2_WGS84 = 6.69437999014e-3  # квадрат эксцентриситета
OMEGA_E = 7.2921151467e-5  # угловая скорость Земли, рад/с
G0 = 9.7803267715  # экваториальная гравитация, м/с²

# Starting point: Kuzbass (Kedrovsky Open-Pit Mine)
INIT_LAT = 55.4400  # deg
INIT_LON = 86.0500  # deg
INIT_ALT = 250.0  # m

# Frequencies
FS_IMU = 400.0  # IMU sample frequency, Hz
FS_REF = 400.0  # Hz, ref freq
FS_GPS = 10

# Динамические ограничения БелАЗ-7513 (гружёный, ~240 т)
# need for mode = [max_acceleration, max_angular_acceleration, max_angular_velocity] [m/s/s, deg/s/s, deg/s]
MAX_ACCEL = 0.5  # m/s^2 — максимальное ускорение
MAX_ACCEL_VERT = 7.3  # deg/s^2 — максимальное угловое ускорение
MAX_JERK = 18.4  # des/s - максимальная угловая скорость
# MAX_ACCEL_LONG = 1.0  # m/s^2 — продольное ускорение


# Other constants
D2R = np.pi / 180
