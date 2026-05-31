import numpy as np

# Реалистичные амплитуды вибраций ДВС карьерного самосвала (гружёного)
VIB_AMPL_LONG = 0.10  # g — продольная (вдоль оси Y по datasheet Н1В)
VIB_AMPL_LAT = 0.05  # g — поперечная
VIB_AMPL_VERT = 0.25  # g — вертикальная (наибольшая для ДВС)

# Гармоники ДВС: 1-я (33.3 Гц), 2-я (66.6 Гц), 3-я (100 Гц)
ENGINE_RPM = 2000.0
F1 = ENGINE_RPM / 60.0  # = 33.33 Гц
HARMONICS = [
    (F1, 1.00, 0.0),  # (частота Гц, относительная амплитуда, фаза)
    (2 * F1, 0.45, np.pi / 4),
    (3 * F1, 0.20, np.pi / 2),
]

# Уширение пика для имитации нестабильности оборотов
RPM_JITTER_STD = 0.05  # 5% разброс частоты вращения


def add_engine_vibration(accel_b, gyro_b, fs, mode="harmonics_plus_noise", seed=42):
    """
    Добавляет вибрации ДВС к сигналам IMU в body-frame.

    mode:
      'single_tone'         — чистая синусоида 33.3 Гц
      'harmonics'           — 3 гармоники без шума
      'harmonics_plus_noise'— 3 гармоники + band-limited шум вокруг каждой
    """
    rng = np.random.default_rng(seed)
    N = accel_b.shape[0]
    t = np.arange(N) / fs

    ampl_xyz_acc = np.array([VIB_AMPL_LAT, VIB_AMPL_LONG, VIB_AMPL_VERT]) * 9.81  # м/с²
    # Гироскопы тоже чувствуют вибрацию (через перекосы осей), но в 50–100 раз слабее
    ampl_xyz_gyr = np.array([0.05, 0.05, 0.02])  # рад/с — небольшая модуляция

    vib_acc = np.zeros_like(accel_b)
    vib_gyr = np.zeros_like(gyro_b)

    for f, rel_a, phi in HARMONICS:
        if mode == "single_tone" and f != F1:
            continue

        if mode == "harmonics_plus_noise":
            # Уширение спектральной линии — модуляция частоты small jitter
            jitter = rng.normal(0, RPM_JITTER_STD, N).cumsum() / fs
            sig = np.sin(2 * np.pi * f * (t + jitter) + phi)
            # Плюс узкополосный шум амплитуды 10% от основной гармоники
            sig += 0.1 * rng.standard_normal(N)
        else:
            sig = np.sin(2 * np.pi * f * t + phi)

        for ax in range(3):
            vib_acc[:, ax] += rel_a * ampl_xyz_acc[ax] * sig
            vib_gyr[:, ax] += rel_a * ampl_xyz_gyr[ax] * sig * 0.5

    return accel_b + vib_acc, gyro_b + vib_gyr
