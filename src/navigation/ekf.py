"""
LC ИНС/СНС: nominal-state EKF с ориентацией в векторе состояния.

ВЕКТОР СОСТОЯНИЯ (внутреннее, 16-D):
    x = [r_n (3), v_n (3), q_bn (4), b_a (3), b_g (3)]^T

Для EKF используется error-state представление 15-D:
    δx = [δr_n (3), δv_n (3), δθ (3), δb_a (3), δb_g (3)]^T,
где δθ — малый поворот, q_bn ← q_bn ⊗ exp(δθ/2).
"""

import numpy as np
from src.config.constants import OMEGA_E, INIT_LAT


def skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def quat_mul(p, q):
    """p ⊗ q, scalar-first."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ]
    )


def quat_to_dcm(q):
    """body → nav DCM."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_rotvec(v):
    """Кватернион из вектора поворота (рад)."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
    half = n / 2.0
    s = np.sin(half) / n
    return np.array([np.cos(half), s * v[0], s * v[1], s * v[2]])


def quat_to_euler_zyx(q):
    """ZYX → [yaw, pitch, roll] (рад)."""
    w, x, y, z = q
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([yaw, pitch, roll])


class InsGnssEKF:
    """Nominal-state EKF для LC интеграции ИНС/СНС."""

    def __init__(
        self,
        fs_imu,
        fs_gps,
        sigma_a_n,
        sigma_g_n,
        sigma_ba,
        sigma_bg,
        tau_ba=500.0,
        tau_bg=300.0,
        sigma_gps_pos=np.array([0.05, 0.05, 0.10]),
        sigma_gps_vel=np.array([0.05, 0.05, 0.05]),
        init_r=np.zeros(3),
        init_v=np.zeros(3),
        init_q=np.array([1.0, 0.0, 0.0, 0.0]),
        init_ba=np.zeros(3),
        init_bg=np.zeros(3),
        lat0_rad=np.deg2rad(INIT_LAT),
    ):
        self.dt = 1.0 / fs_imu
        self.lat0 = lat0_rad
        self.tau_ba = tau_ba
        self.tau_bg = tau_bg

        self.omega_ie_n = np.array(
            [OMEGA_E * np.cos(lat0_rad), 0.0, -OMEGA_E * np.sin(lat0_rad)]
        )

        sl2 = np.sin(lat0_rad) ** 2
        g_mag = 9.7803267715 * (1 + 0.0052790414 * sl2 + 0.0000232718 * sl2**2)
        self.g_n = np.array([0.0, 0.0, g_mag])  # NED: +z вниз

        # Nominal state
        self.r = init_r.astype(float).copy()
        self.v = init_v.astype(float).copy()
        self.q = init_q.astype(float).copy()
        self.q /= np.linalg.norm(self.q)
        self.b_a = init_ba.astype(float).copy()
        self.b_g = init_bg.astype(float).copy()

        # P 15×15
        self.P = np.diag(
            [
                *(np.array([1.0, 1.0, 2.0]) ** 2),
                *(np.array([0.1, 0.1, 0.1]) ** 2),
                *(np.deg2rad([1.0, 1.0, 5.0]) ** 2),
                *(sigma_ba * 10) ** 2,
                *(sigma_bg * 10) ** 2,
            ]
        )

        # Q непрерывный
        self.Q_c = np.diag(
            [
                0,
                0,
                0,
                *(sigma_a_n**2),
                *(sigma_g_n**2),
                *(sigma_ba**2),
                *(sigma_bg**2),
            ]
        )

        # R, H
        self.R = np.diag(np.concatenate([sigma_gps_pos**2, sigma_gps_vel**2]))
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        self.H = H

    # Априорная оценка
    def predict(self, accel_b, gyro_b):
        dt = self.dt
        a_b = accel_b - self.b_a
        w_b = gyro_b - self.b_g

        C_bn = quat_to_dcm(self.q)
        f_n = C_bn @ a_b
        a_n = f_n + self.g_n - 2.0 * np.cross(self.omega_ie_n, self.v)

        v_new = self.v + a_n * dt
        self.r = self.r + 0.5 * (self.v + v_new) * dt
        self.v = v_new

        # Интегрирование кватерниона: ω_nb_b = ω_ib_b - C_bn^T · ω_ie_n
        w_nb_b = w_b - C_bn.T @ self.omega_ie_n
        self.q = quat_mul(self.q, quat_from_rotvec(w_nb_b * dt))
        self.q /= np.linalg.norm(self.q)

        # Распространение ковариации
        F = self._build_F(f_n, C_bn)
        Phi = np.eye(15) + F * dt
        self.P = Phi @ self.P @ Phi.T + self.Q_c * dt

    def _build_F(self, f_n, C_bn):
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 3:6] = -2.0 * skew(self.omega_ie_n)
        F[3:6, 6:9] = -skew(f_n)
        F[3:6, 9:12] = C_bn
        F[6:9, 6:9] = -skew(self.omega_ie_n)
        F[6:9, 12:15] = -C_bn
        F[9:12, 9:12] = -np.eye(3) / self.tau_ba
        F[12:15, 12:15] = -np.eye(3) / self.tau_bg
        return F

    # Апостериорная оценка
    def update_gnss(self, r_gps_ned, v_gps_ned):
        z = np.concatenate([r_gps_ned - self.r, v_gps_ned - self.v])
        H, R, P = self.H, self.R, self.P
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        dx = K @ z
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self._inject(dx)

    def _inject(self, dx):
        self.r += dx[0:3]
        self.v += dx[3:6]
        self.q = quat_mul(self.q, quat_from_rotvec(dx[6:9]))
        self.q /= np.linalg.norm(self.q)
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

    @property
    def euler_zyx_deg(self):
        return np.rad2deg(quat_to_euler_zyx(self.q))
