import numpy as np
from src.config.constants import INIT_LAT

import gtsam

from gtsam import (
    NonlinearFactorGraph,
    Values,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    PreintegratedImuMeasurements,
    ImuFactor,
    GPSFactor,
    Pose3,
    Rot3,
    Point3,
    NavState,
    imuBias,
    PriorFactorPose3,
    PriorFactorVector,
    PriorFactorConstantBias,
    BetweenFactorConstantBias,
)

from gtsam.symbol_shorthand import X, V, B  # X=поза, V=скорость, B=смещения


class InsGnssFGO:
    """FGO для LC интеграции ИНС/СНС на GTSAM."""

    def __init__(
        self,
        fs_imu,
        sigma_a_n,
        sigma_g_n,
        sigma_ba,
        sigma_bg,
        sigma_gps_pos=np.array([0.05, 0.05, 0.10]),
        init_pose=None,
        init_vel=np.zeros(3),
        init_ba=np.zeros(3),
        init_bg=np.zeros(3),
        lat0_rad=np.deg2rad(INIT_LAT),
        window_sec=None,
        fs_gps=10.0,
    ):
        self.dt = 1.0 / fs_imu
        # размер окна оптимизации в узлах GNSS (0/None -> весь граф батчем)
        self.window_nodes = int(window_sec * fs_gps) if window_sec else None

        sl2 = np.sin(lat0_rad) ** 2
        g_mag = 9.7803267715 * (1 + 0.0052790414 * sl2 + 0.0000232718 * sl2**2)

        # Параметры предынтегрирования (NED: гравитация +z вниз)
        params = gtsam.PreintegrationParams.MakeSharedD(g_mag)
        params.setAccelerometerCovariance(np.eye(3) * sigma_a_n[0] ** 2)
        params.setGyroscopeCovariance(np.eye(3) * sigma_g_n[0] ** 2)
        params.setIntegrationCovariance(np.eye(3) * 1e-8)
        self.params = params

        self.bias = imuBias.ConstantBias(init_ba, init_bg)
        self.pim = PreintegratedImuMeasurements(params, self.bias)

        # Граф и начальные значения
        self.graph = NonlinearFactorGraph()
        self.values = Values()

        # Начальное состояние (узел 0)
        if init_pose is None:
            init_pose = Pose3(Rot3.Identity(), Point3(0, 0, 0))
        self.values.insert(X(0), init_pose)
        self.values.insert(V(0), init_vel)
        self.values.insert(B(0), self.bias)

        # Априорные факторы на узел 0
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.1, 1.0, 1.0, 2.0])
        )  # rot(3)+pos(3)
        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.graph.add(PriorFactorPose3(X(0), init_pose, pose_noise))
        self.graph.add(PriorFactorVector(V(0), init_vel, vel_noise))
        self.graph.add(PriorFactorConstantBias(B(0), self.bias, bias_noise))

        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(sigma_gps_pos)
        self.bias_between_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, np.sqrt(self.dt) * (sigma_ba[0] + sigma_bg[0])
        )

        self.key = 0  # текущий индекс узла (на эпохах GNSS)
        self._prev_state = NavState(init_pose, init_vel)

    def integrate(self, accel_b, gyro_b):
        """Накопить одно измерение IMU в предынтегратор (вызывать @ fs_imu)."""
        self.pim.integrateMeasurement(accel_b, gyro_b, self.dt)

    def add_gnss(self, r_gps_ned):
        """
        Закрыть текущий интервал предынтегрирования новым узлом и добавить
        GNSS-фактор. Вызывать на каждой эпохе GNSS.
        """
        k0, k1 = self.key, self.key + 1
        # IMU-фактор между k0 и k1
        self.graph.add(ImuFactor(X(k0), V(k0), X(k1), V(k1), B(k0), self.pim))
        # эволюция смещений (case-to-case Гаусс-Марков ~ between)
        self.graph.add(
            BetweenFactorConstantBias(
                B(k0), B(k1), imuBias.ConstantBias(), self.bias_between_noise
            )
        )
        # GNSS-фактор на позицию нового узла
        self.graph.add(GPSFactor(X(k1), Point3(*r_gps_ned), self.gps_noise))

        # Начальная догадка для нового узла — прогноз предынтегратора
        pred = self.pim.predict(self._prev_state, self.bias)
        self.values.insert(X(k1), pred.pose())
        self.values.insert(V(k1), pred.velocity())
        self.values.insert(B(k1), self.bias)

        self._prev_state = pred
        self.key = k1
        # сброс предынтегратора под новый интервал
        self.pim.resetIntegrationAndSetBias(self.bias)

    def optimize(self):
        """Решить граф. Если задано окно (window_nodes) — решаем скользящими
        кусками по window_nodes узлов, фиксируя стык prior'ом из предыдущего
        куска (склейка по непрерывности). Иначе — весь граф батчем (LM)."""
        params = LevenbergMarquardtParams()
        params.setMaxIterations(100)

        n = self.key + 1
        if not self.window_nodes or self.window_nodes >= n:
            # батч: весь граф разом
            opt = LevenbergMarquardtOptimizer(self.graph, self.values, params)
            result = opt.optimize()
            return self._extract(result, n)

        # оконная оптимизация: куски по window_nodes с перекрытием в 1 узел
        W = self.window_nodes
        R = np.zeros((n, 3)); V_ = np.zeros((n, 3)); E = np.zeros((n, 3))

        # все факторы заранее не делим — строим под-граф для каждого окна,
        # переиспользуя уже добавленные факторы по диапазону ключей.
        all_factors = [self.graph.at(i) for i in range(self.graph.size())]

        start = 0
        carry_pose = None
        carry_vel = None
        carry_bias = None
        while start < n:
            end = min(start + W, n)  # узлы [start, end)
            sub = NonlinearFactorGraph()
            keys_in = set(range(start, end))
            for f in all_factors:
                # фактор включаем, если все его навигационные ключи в окне
                fk = f.keys()
                idxs = [gtsam.symbolIndex(k) for k in fk]
                if all((i in keys_in) for i in idxs):
                    sub.add(f)

            sub_vals = Values()
            for i in range(start, end):
                sub_vals.insert(X(i), self.values.atPose3(X(i)))
                sub_vals.insert(V(i), self.values.atVector(V(i)))
                sub_vals.insert(B(i), self.values.atConstantBias(B(i)))

            # prior на первый узел окна — из переноса предыдущего решения
            if carry_pose is not None:
                pn = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([0.01, 0.01, 0.05, 0.3, 0.3, 0.5]))
                vn = gtsam.noiseModel.Isotropic.Sigma(3, 0.05)
                sub.add(PriorFactorPose3(X(start), carry_pose, pn))
                sub.add(PriorFactorVector(V(start), carry_vel, vn))
                sub.add(PriorFactorConstantBias(
                    B(start), carry_bias,
                    gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))

            opt = LevenbergMarquardtOptimizer(sub, sub_vals, params)
            res = opt.optimize()

            for i in range(start, end):
                pose = res.atPose3(X(i))
                R[i] = pose.translation()
                V_[i] = res.atVector(V(i))
                rpy = pose.rotation().rpy()
                E[i] = np.rad2deg([rpy[2], rpy[1], rpy[0]])

            # перенос последнего узла как prior следующего окна
            carry_pose = res.atPose3(X(end - 1))
            carry_vel = res.atVector(V(end - 1))
            carry_bias = res.atConstantBias(B(end - 1))
            if end >= n:
                break
            start = end - 1  # перекрытие в 1 узел для непрерывности

        return R, V_, E

    def _extract(self, result, n):
        R = np.zeros((n, 3)); V_ = np.zeros((n, 3)); E = np.zeros((n, 3))
        for i in range(n):
            pose = result.atPose3(X(i))
            R[i] = pose.translation()
            V_[i] = result.atVector(V(i))
            rpy = pose.rotation().rpy()
            E[i] = np.rad2deg([rpy[2], rpy[1], rpy[0]])
        self.result = result
        return R, V_, E