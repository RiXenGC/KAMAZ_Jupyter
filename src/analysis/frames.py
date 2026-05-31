import numpy as np

from src.simulation.run_simulation import run_simulation
from src.simulation.data_loader import load_data
from src.simulation.vibration import add_engine_vibration
from src.config.constants import A_WGS84
from src.config.config import IMU_ERR, GPS_ERR


def euler_to_quat_zyx(yaw, pitch, roll):
    """ZYX Эйлер (рад) → кватернион [w,x,y,z]."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def lla_to_ned(lat, lon, h, lat0, lon0, h0):
    """LLA (рад/рад/м) → локальные NED-метры от точки старта."""
    return np.column_stack(
        [
            (lat - lat0) * A_WGS84,  # North
            (lon - lon0) * A_WGS84 * np.cos(lat0),  # East
            -(h - h0),  # Down
        ]
    )


def prepare_scenario(
    scenario, motion_dir, data_dir, vibration_mode="harmonics_plus_noise", run_sim=True
):
    """
    Полная подготовка одного сценария: прогон симулятора (опц.),
    загрузка, перевод в NED, шумы, вибрация.

    Возвращает dict со всем, что нужно фильтрам.
    """

    motion_csv = f"{motion_dir}/{scenario}.csv"
    out_dir = f"{data_dir}/{scenario}"

    if run_sim:
        run_simulation(motion_csv, out_dir)
    data = load_data(out_dir)

    # --- IMU ---
    accel = np.asarray(data["accel"])
    gyro = np.asarray(data["gyro"])
    imu_time = np.asarray(data["imu_time"])

    # --- точка старта ---
    lat0 = np.deg2rad(data["ref_pos"][0, 0])
    lon0 = np.deg2rad(data["ref_pos"][0, 1])
    h0 = data["ref_pos"][0, 2]

    # --- reference в NED ---
    lat = np.deg2rad(data["ref_pos"][:, 0])
    lon = np.deg2rad(data["ref_pos"][:, 1])
    h = data["ref_pos"][:, 2]
    ref_pos_ned = lla_to_ned(lat, lon, h, lat0, lon0, h0)
    ref_vel_ned = np.asarray(data["ref_vel"])
    ref_euler_ned = np.deg2rad(np.asarray(data["ref_att"]))

    # --- GPS в NED ---
    gps_lla = np.asarray(data["gps"][:, 0:3]).copy()
    gps_lla[:, 0:2] = np.deg2rad(gps_lla[:, 0:2])
    gps_pos_ned = lla_to_ned(
        gps_lla[:, 0], gps_lla[:, 1], gps_lla[:, 2], lat0, lon0, h0
    )
    gps_vel_ned = np.asarray(data["gps"][:, 3:6])
    gps_idx = np.searchsorted(imu_time, data["gps_time"])

    # --- начальный кватернион ---
    q0 = euler_to_quat_zyx(*ref_euler_ned[0])

    # --- шумы ---
    noise = dict(
        sigma_a_n=IMU_ERR["accel_vrw"] / 60.0,
        sigma_g_n=np.deg2rad(IMU_ERR["gyro_arw"]) / 60.0,
        sigma_ba=IMU_ERR["accel_b_stability"],
        sigma_bg=np.deg2rad(IMU_ERR["gyro_b_stability"] / 3600.0),
        sigma_gps_pos=GPS_ERR["stdp"],
        sigma_gps_vel=GPS_ERR["stdv"],
    )

    # --- вибрация ---
    accel_vib, gyro_vib = add_engine_vibration(accel, gyro, fs=400, mode=vibration_mode)

    return {
        "accel": accel,
        "gyro": gyro,
        "imu_time": imu_time,
        "accel_vib": accel_vib,
        "gyro_vib": gyro_vib,
        "ref_pos_ned": ref_pos_ned,
        "ref_vel_ned": ref_vel_ned,
        "ref_euler_ned": ref_euler_ned,
        "gps_pos_ned": gps_pos_ned,
        "gps_vel_ned": gps_vel_ned,
        "gps_idx": gps_idx,
        "q0": q0,
        "lat0_deg": data["ref_pos"][0, 0],
        "noise": noise,
    }
