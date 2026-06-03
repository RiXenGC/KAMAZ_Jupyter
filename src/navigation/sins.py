import numpy as np
from scipy.integrate import cumulative_trapezoid
from ahrs.filters import Madgwick, Mahony
from ahrs.common.orientation import q2R

from src.config.constants import G0, OMEGA_E, INIT_LAT

# ahrs.Madgwick/Mahony работают в ENU/FLU: acc = [0,0,+1].
# gnss-ins-sim работает в NED/FRD: acc = [0,0,-9.8].
T_BODY = np.diag([1.0, -1.0, -1.0])  # FRD <-> FLU (body)
T_NAV = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]  # ENU <-> NED (nav)
)


def gravity_ned(lat_rad, h):
    sl2 = np.sin(lat_rad) ** 2
    g = G0 * (1 + 0.0052790414 * sl2 + 0.0000232718 * sl2**2) - 3.087e-6 * h
    return np.array([0.0, 0.0, g])


def earth_rate_ned(lat_rad):
    return np.array([OMEGA_E * np.cos(lat_rad), 0.0, -OMEGA_E * np.sin(lat_rad)])


def _R_to_quat(R):

    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _flip_quat(q):
    """Кватернион NED<->ENU: R' = T_nav · R · T_body."""
    R = q2R(q)
    R2 = T_NAV @ R @ T_BODY
    return _R_to_quat(R2)


class SINSBase:
    def __init__(
        self,
        fs,
        init_pos_ned=np.zeros(3),
        init_vel_ned=np.zeros(3),
        init_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        lat0_deg=INIT_LAT,
    ):
        self.fs = fs
        self.dt = 1.0 / fs
        self.lat0 = np.deg2rad(lat0_deg)
        self.q0 = init_quat / np.linalg.norm(init_quat)  # NED (FRD->NED)
        self.p0 = init_pos_ned.copy()
        self.v0 = init_vel_ned.copy()

    def estimate_attitude(self, accel_b, gyro_b, q_init):
        raise NotImplementedError("Реализуйте в подклассе")

    def specific_force_to_nframe(self, accel_b, Q):
        N = accel_b.shape[0]
        f_n = np.zeros_like(accel_b)
        for k in range(N):
            R = q2R(Q[k])
            f_n[k] = R @ accel_b[k]
        return f_n

    def mechanize(self, f_n):
        N = f_n.shape[0]
        t = np.arange(N) * self.dt
        g_n = gravity_ned(self.lat0, 0.0)
        omega_ie = earth_rate_ned(self.lat0)

        v = np.zeros((N, 3))
        v[0] = self.v0
        a = np.zeros((N, 3))
        a[0] = f_n[0] + g_n - 2 * np.cross(omega_ie, v[0])

        for k in range(1, N):
            a_k = f_n[k] + g_n - 2 * np.cross(omega_ie, v[k - 1])
            a[k] = a_k
            v[k] = v[k - 1] + 0.5 * (a[k - 1] + a_k) * self.dt

        p = self.p0 + cumulative_trapezoid(v, t, axis=0, initial=0)
        return a, v, p

    def _quat_to_euler_zyx(self, Q):
        """ZYX-разложение кватерниона в [yaw, pitch, roll] (град). Конвенция = конвенция Q."""
        N = Q.shape[0]
        eul = np.zeros((N, 3))
        for k in range(N):
            w, x, y, z = Q[k]
            sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
            pitch = np.arcsin(sinp)
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            eul[k] = [np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)]
        return eul

    def _euler_enu_to_ned(self, eul_enu):
        """[yaw,pitch,roll] ENU -> NED: yaw_ned=90-yaw_enu, pitch меняет знак, roll без изменений."""
        eul = np.empty_like(eul_enu)
        yaw_ned = (90.0 - eul_enu[:, 0]) % 360.0
        yaw_ned = np.where(yaw_ned > 180.0, yaw_ned - 360.0, yaw_ned)
        eul[:, 0] = yaw_ned
        eul[:, 1] = -eul_enu[:, 1]
        eul[:, 2] = eul_enu[:, 2]
        return eul

    def run(self, accel_b, gyro_b):

        # FRD -> FLU
        accel_flu = accel_b @ T_BODY
        gyro_flu = gyro_b @ T_BODY

        # NED -> ENU для фильтра
        q0_enu = _flip_quat(self.q0)

        Q_enu = self.estimate_attitude(accel_flu, gyro_flu, q0_enu)

        # Specific force ENU -> NED
        f_n_enu = self.specific_force_to_nframe(accel_flu, Q_enu)
        f_n = f_n_enu @ T_NAV

        # Mechanization NED
        a_n, v_n, p_n = self.mechanize(f_n)

        # ориентация на выход — в NED
        euler = self._euler_enu_to_ned(self._quat_to_euler_zyx(Q_enu))
        Q_ned = np.array([_flip_quat(q) for q in Q_enu])

        self.q_hist = Q_ned
        self.f_n_hist = f_n
        self.v_hist = v_n
        self.p_hist = p_n
        self.euler_hist = euler

        return {"q_NED": Q_ned, "v_n": v_n, "p_n": p_n, "euler": euler}


# Orientation filters
class SINSMadgwick(SINSBase):

    def __init__(self, fs, beta=0.03, **kwargs):
        super().__init__(fs, **kwargs)
        self.beta = beta
        self.filter = Madgwick(frequency=fs, beta=beta)

    def estimate_attitude(self, accel_b, gyro_b, q_init):
        N = accel_b.shape[0]
        Q = np.zeros((N, 4))
        Q[0] = q_init
        for k in range(1, N):
            Q[k] = self.filter.updateIMU(Q[k - 1], gyr=gyro_b[k], acc=accel_b[k])
        return Q


class SINSMahony(SINSBase):

    def __init__(self, fs, k_P=0.1, k_I=0.001, **kwargs):
        super().__init__(fs, **kwargs)
        self.k_P = k_P
        self.k_I = k_I
        self.filter = Mahony(frequency=fs, k_P=k_P, k_I=k_I)

    def estimate_attitude(self, accel_b, gyro_b, q_init):
        N = accel_b.shape[0]
        Q = np.zeros((N, 4))
        Q[0] = q_init
        for k in range(1, N):
            Q[k] = self.filter.updateIMU(Q[k - 1], gyr=gyro_b[k], acc=accel_b[k])
        return Q


# Select filter
def make_sins(filter_name, fs, **kwargs):

    if filter_name.lower() == "madgwick":
        return SINSMadgwick(fs, **kwargs)
    elif filter_name.lower() == "mahony":
        return SINSMahony(fs, **kwargs)
    else:
        raise ValueError(f"Неизвестный фильтр: {filter_name}")
