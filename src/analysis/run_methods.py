import numpy as np
from scipy.interpolate import interp1d

from src.navigation.ekf import InsGnssEKF
from src.navigation.ukf import InsGnssUKF
from src.navigation.fgo import InsGnssFGO

import gtsam
from gtsam import Pose3, Rot3, Point3


def run_kalman(filter_cls, S, accel, gyro):
    n = S["noise"]
    filt = filter_cls(
        lat0_rad=np.deg2rad(S["lat0_deg"]),
        fs_imu=400.0,
        fs_gps=10.0,
        sigma_a_n=n["sigma_a_n"],
        sigma_g_n=n["sigma_g_n"],
        sigma_ba=n["sigma_ba"],
        sigma_bg=n["sigma_bg"],
        sigma_gps_pos=n["sigma_gps_pos"],
        sigma_gps_vel=n["sigma_gps_vel"],
        init_r=S["ref_pos_ned"][0],
        init_v=S["ref_vel_ned"][0],
        init_q=S["q0"],
    )
    N = len(S["imu_time"])

    p, v, eul = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))

    g_ptr = 0
    gps_idx = S["gps_idx"]
    for k in range(N):
        filt.predict(accel[k], gyro[k])
        if g_ptr < len(gps_idx) and k == gps_idx[g_ptr]:
            filt.update_gnss(S["gps_pos_ned"][g_ptr], S["gps_vel_ned"][g_ptr])
            g_ptr += 1
        p[k] = filt.r
        v[k] = filt.v
        eul[k] = filt.euler_zyx_deg
    return p, v, eul


def _interp_to_imu(arr_nodes, t_nodes, t_imu):
    """Интерполяция узлов FGO на сетку IMU"""
    t_nodes = np.asarray(t_nodes, dtype=float)
    arr_nodes = np.asarray(arr_nodes, dtype=float)

    keep = np.concatenate([[True], np.diff(t_nodes) > 1e-9])

    t_nodes, arr_nodes = t_nodes[keep], arr_nodes[keep]
    f = interp1d(
        t_nodes,
        arr_nodes,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(arr_nodes[0], arr_nodes[-1]),
    )

    out = f(t_imu)
    if np.isnan(out).any():
        for j in range(out.shape[1]):
            col = out[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.interp(
                    np.flatnonzero(mask), np.flatnonzero(~mask), col[~mask]
                )
    return out


def run_fgo(S, accel, gyro):

    n = S["noise"]
    w, x, y, z = S["q0"]
    init_pose = Pose3(Rot3.Quaternion(w, x, y, z), Point3(*S["ref_pos_ned"][0]))
    fgo = InsGnssFGO(
        fs_imu=400.0,
        sigma_a_n=n["sigma_a_n"],
        sigma_g_n=n["sigma_g_n"],
        sigma_ba=n["sigma_ba"],
        sigma_bg=n["sigma_bg"],
        sigma_gps_pos=n["sigma_gps_pos"],
        init_pose=init_pose,
        init_vel=S["ref_vel_ned"][0],
    )
    N = len(S["imu_time"])
    g_ptr = 0
    gps_idx = S["gps_idx"]
    node_t = [S["imu_time"][0]]
    for k in range(N):
        fgo.integrate(accel[k], gyro[k])
        if g_ptr < len(gps_idx) and k == gps_idx[g_ptr]:
            fgo.add_gnss(S["gps_pos_ned"][g_ptr])
            node_t.append(S["imu_time"][k])
            g_ptr += 1
    p_nodes, v_nodes, eul_nodes = fgo.optimize()
    node_t = np.array(node_t)

    t_imu = S["imu_time"]
    p = _interp_to_imu(p_nodes, node_t, t_imu)
    v = _interp_to_imu(v_nodes, node_t, t_imu)
    eul = _interp_to_imu(eul_nodes, node_t, t_imu)
    return p, v, eul


def run_all_methods(S, use_vibration=False):
    if use_vibration:
        accel, gyro = S["accel_vib"], S["gyro_vib"]
    else:
        accel, gyro = S["accel"], S["gyro"]

    p_ekf, v_ekf, e_ekf = run_kalman(InsGnssEKF, S, accel, gyro)
    p_ukf, v_ukf, e_ukf = run_kalman(InsGnssUKF, S, accel, gyro)
    p_fgo, v_fgo, e_fgo = run_fgo(S, accel, gyro)

    return {
        "EKF": {"pos": p_ekf, "vel": v_ekf, "euler": e_ekf},
        "UKF": {"pos": p_ukf, "vel": v_ukf, "euler": e_ukf},
        "FGO": {"pos": p_fgo, "vel": v_fgo, "euler": e_fgo},
    }
