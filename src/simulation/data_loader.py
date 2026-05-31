import pandas as pd
from pathlib import Path


def load_data(sim_dir):
    """Загружает данные из выхода gnss-ins-sim."""
    sim_dir = Path(sim_dir)
    # Находим подпапку с timestamp
    subdirs = [d for d in sim_dir.iterdir() if d.is_dir()]
    data_dir = subdirs[0] if subdirs else sim_dir

    def _read(name):
        """Безопасное чтение CSV: None, если файла нет."""
        path = data_dir / name
        return pd.read_csv(path).values if path.exists() else None

    # Output data for IMU (FS_IMU)
    imu_time = _read("time.csv").flatten()
    accel = _read("accel-0.csv")  # m/s^2
    gyro = _read("gyro-0.csv")  # rad/s

    # Output data for GPS (FS_GPS)
    gps = _read("gps-0.csv")  # [pos(3), vel(3)]
    gps_time = _read("gps_time.csv")
    if gps_time is not None:
        gps_time = gps_time.flatten()

    # Reference data
    ref_pos = _read("ref_pos.csv")
    ref_vel = _read("ref_vel.csv")
    ref_att = _read("ref_att_euler.csv")  # deg

    return {
        "accel": accel,
        "gyro": gyro,
        "imu_time": imu_time,
        "gps": gps,
        "gps_time": gps_time,
        "ref_pos": ref_pos,
        "ref_vel": ref_vel,
        "ref_att": ref_att,
    }
