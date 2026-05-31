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
# need for mode = [m/s/s, deg/s/s, deg/s]
MAX_ACCEL_LONG = 1.0  # m/s^2 — продольное ускорение
MAX_ACCEL_LAT = 0.8  # m/s^2 — поперечное при повороте
MAX_ACCEL_VERT = 0.5  # m/s^2 — вертикальное
MAX_JERK = 0.3  # m/s^3


# Other constants
D2R = 3.14 / 180
G = 9.81
