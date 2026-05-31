"""
LC ИНС/СНС: UKF с ориентацией в векторе состояния.

Та же модель, что и InsGnssEKF, но прогноз и обновление через unscented
преобразование. Использует сигма-точки (Julier–Uhlmann) для нелинейного
распространения распределения.

Используем filterpy.kalman.UnscentedKalmanFilter с MerweScaledSigmaPoints.
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from src.navigation.ekf import (
    OMEGA_E,
    skew,
    quat_mul,
    quat_to_dcm,
    quat_from_rotvec,
    quat_to_euler_zyx,
)


class InsGnssUKF:
    """UKF для LC интеграции ИНС/СНС. State: [r, v, q, b_a, b_g] = 16-D."""

    def __init__(
        self,
        lat0_rad,
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
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
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
        self.g_n = np.array([0.0, 0.0, g_mag])

        n = 16
        points = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        self.ukf = UnscentedKalmanFilter(
            dim_x=n,
            dim_z=6,
            dt=self.dt,
            fx=self._fx,
            hx=self._hx,
            points=points,
        )

        # Текущие IMU-измерения (передаются в fx как замыкание)
        self._a_b = np.zeros(3)
        self._w_b = np.zeros(3)

        # Начальное состояние
        q0 = init_q / np.linalg.norm(init_q)
        x0 = np.concatenate([init_r, init_v, q0, init_ba, init_bg])
        self.ukf.x = x0

        # Ковариация (для UKF удобно работать в "развёрнутом" 16-D,
        # с пониманием что 4 компоненты кватерниона коррелированы)
        P0 = np.diag(
            [
                *(np.array([1.0, 1.0, 2.0]) ** 2),
                *(np.array([0.1, 0.1, 0.1]) ** 2),
                *(np.array([0.01] * 4) ** 2),  # δq порядка 1° → 0.01
                *(sigma_ba * 10) ** 2,
                *(sigma_bg * 10) ** 2,
            ]
        )
        self.ukf.P = P0

        # Q в 16-D (на компоненты, не имеющие шума, ставим малое)
        self.ukf.Q = np.diag(
            [
                *(1e-12 * np.ones(3)),  # r
                *(sigma_a_n**2 * self.dt),  # v
                *(1e-10 * np.ones(4)),  # q малый
                *(sigma_ba**2 * self.dt),
                *(sigma_bg**2 * self.dt),
            ]
        )
        # gyroscope noise входит через q-канал — увеличим q-Q
        self.ukf.Q[6:10, 6:10] += (sigma_g_n[0] ** 2 * self.dt) * np.eye(4) * 0.25

        # R по GNSS
        self.ukf.R = np.diag(np.concatenate([sigma_gps_pos**2, sigma_gps_vel**2]))

    # Динамика для UKF (16-D)
    def _fx(self, x, dt):
        r = x[0:3]
        v = x[3:6]
        q = x[6:10]
        ba = x[10:13]
        bg = x[13:16]
        q = q / np.linalg.norm(q)

        a_b = self._a_b - ba
        w_b = self._w_b - bg
        C_bn = quat_to_dcm(q)
        f_n = C_bn @ a_b
        a_n = f_n + self.g_n - 2.0 * np.cross(self.omega_ie_n, v)

        v_new = v + a_n * dt
        r_new = r + 0.5 * (v + v_new) * dt
        w_nb_b = w_b - C_bn.T @ self.omega_ie_n
        q_new = quat_mul(q, quat_from_rotvec(w_nb_b * dt))
        q_new = q_new / np.linalg.norm(q_new)
        # Гаусс-Марков для смещений
        ba_new = ba - (dt / self.tau_ba) * ba
        bg_new = bg - (dt / self.tau_bg) * bg
        return np.concatenate([r_new, v_new, q_new, ba_new, bg_new])

    # Измерения (LC GNSS): прямо отдают позицию и скорость
    def _hx(self, x):
        return np.concatenate([x[0:3], x[3:6]])

    # ---------- ПУБЛИЧНЫЙ ИНТЕРФЕЙС ----------
    def predict(self, accel_b, gyro_b):
        self._a_b = accel_b
        self._w_b = gyro_b
        self.ukf.predict()
        # Перенормировать кватернион в state
        self.ukf.x[6:10] /= np.linalg.norm(self.ukf.x[6:10])

    def update_gnss(self, r_gps_ned, v_gps_ned):
        z = np.concatenate([r_gps_ned, v_gps_ned])
        self.ukf.update(z)
        self.ukf.x[6:10] /= np.linalg.norm(self.ukf.x[6:10])

    # Удобство
    @property
    def r(self):
        return self.ukf.x[0:3]

    @property
    def v(self):
        return self.ukf.x[3:6]

    @property
    def q(self):
        return self.ukf.x[6:10]

    @property
    def b_a(self):
        return self.ukf.x[10:13]

    @property
    def b_g(self):
        return self.ukf.x[13:16]

    @property
    def euler_zyx_deg(self):
        return np.rad2deg(quat_to_euler_zyx(self.q))
