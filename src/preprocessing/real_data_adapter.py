"""
real_data_adapter.py — сборка словаря сценария S из реального parquet-сегмента
для фильтров ИНС/СНС (EKF / UKF / FGO). Инициализация из первой точки (без эталона).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# -----//----- Конфигурация -----//-----
from src.config.config import IMU_ERR, GPS_ERR
from src.config.constants import A_WGS84, KMH_TO_MS


# -----//----- Вспомогательные -----//-----
def _lla_to_ned(lat_deg, lon_deg, h, lat0_deg, lon0_deg, h0):
    """LLA (град/град/м) -> NED-метры [North, East, Down] от опорной точки.
    Та же конвенция, что в src/analysis/frames.lla_to_ned (NED, Down = -(h-h0))."""
    lat = np.deg2rad(np.asarray(lat_deg, float))
    lon = np.deg2rad(np.asarray(lon_deg, float))
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    north = (lat - lat0) * A_WGS84
    east = (lon - lon0) * A_WGS84 * np.cos(lat0)
    down = -(np.asarray(h, float) - h0)
    return np.column_stack([north, east, down])


def _load_parquet(seg_dir: Path, name: str):
    p = seg_dir / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _lowpass(x, fs, fc, order=4):
    """Нуль-фазовый ФНЧ Баттерворта по столбцам (filtfilt — без задержки).
    fc — частота среза, Гц. Снимает вибрацию ДВС, не трогая медленную динамику."""
    from scipy.signal import butter, filtfilt
    if fc is None or fc <= 0:
        return x
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    pad = 3 * max(len(a), len(b))
    if x.shape[0] <= pad:        # слишком короткий сегмент — не фильтруем
        return x
    return filtfilt(b, a, x, axis=0)


def _gravity_alignment_q(accel, n_samples):
    """Начальный кватернион [w,x,y,z] из specific force на покое.
    accel здесь — удельная сила (на покое ~ [0,0,-g]); крен/тангаж из
    направления гравитации (= -accel), рысканье = 0 (курса нет)."""
    a = accel[:max(1, n_samples)].mean(axis=0)
    a = a / (np.linalg.norm(a) + 1e-12)
    gdir = -a                                # направление гравитации
    pitch = np.arcsin(-gdir[0])
    roll = np.arctan2(gdir[1], gdir[2])
    yaw = 0.0
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# -----//----- Сборка сценария -----//-----
def prepare_real_segment(
    seg_dir,
    gnss_name: str = "gnss0",
    vel_name: str = "vel0",
    lowpass_fc: float | None = 5.0,
    align_init: bool = True,
    align_sec: float = 2.0,
) -> dict:
    """Собирает словарь сценария S из одного размеченного сегмента (parquet).

    Параметры
    ---------
    seg_dir : путь к папке сегмента (содержит gnss0/gnss1/vel0/vel1/imu .parquet)
    gnss_name : какой ГНСС-приёмник использовать как источник позиции
    vel_name  : какой VEL-приёмник использовать как источник скорости

    Возвращает
    ----------
    Словарь S, совместимый с run_methods.run_kalman / run_fgo.
    Дополнительно содержит 'gps_status' (RTK fix-флаг) для отчётов.
    """
    seg_dir = Path(seg_dir)
    imu = _load_parquet(seg_dir, "imu")
    gnss = _load_parquet(seg_dir, gnss_name)
    vel = _load_parquet(seg_dir, vel_name)
    if imu is None or gnss is None or vel is None:
        raise FileNotFoundError(
            f"В {seg_dir} нет нужных parquet (imu/{gnss_name}/{vel_name})."
        )

    # -----//----- IMU (NED-body, рад/с и м/с²) -----//-----
    imu_time = imu["t"].to_numpy(float)
    accel = imu[
        ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
    ].to_numpy(float)
    gyro = imu[
        ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
    ].to_numpy(float)

    # ФНЧ: снимаем вибрацию ДВС (~33 Гц), оставляем динамику (< 2 Гц)
    fs_imu = 1.0 / np.median(np.diff(imu_time)) if len(imu_time) > 1 else 400.0
    accel = _lowpass(accel, fs_imu, lowpass_fc)
    gyro = _lowpass(gyro, fs_imu, lowpass_fc)

    # Конвенция удельной силы: фильтр (как симулятор GNSS-INS-SIM) ждёт от
    # акселерометра specific force — на покое [0,0,-g]. Наш препроцесс даёт
    # measured acceleration [0,0,+g] (гравитация в +Z). Инвертируем знак,
    # иначе остаток гравитации интегрируется в скорость (взлёт до десятков м/с).
    accel = -accel

    # -----//----- GNSS позиция → NED от первой точки -----//-----
    lat0 = float(gnss["latitude"].iloc[0])
    lon0 = float(gnss["longitude"].iloc[0])
    h0 = float(gnss["altitude"].iloc[0])
    gps_pos_ned = _lla_to_ned(
        gnss["latitude"].to_numpy(), gnss["longitude"].to_numpy(),
        gnss["altitude"].to_numpy(), lat0, lon0, h0,
    )
    gps_time = gnss["t"].to_numpy(float)
    gps_status = gnss["status"].to_numpy() if "status" in gnss.columns else \
        np.full(len(gnss), RTK_FIX_STATUS)

    # -----//----- VEL: км/ч → м/с, Down = 0 -----//-----
    # vel и gnss идут на одной 10-Гц сетке (preprocess), длины совпадают.
    v_n = vel["v_x_kmh"].to_numpy(float) * KMH_TO_MS  # North
    v_e = vel["v_y_kmh"].to_numpy(float) * KMH_TO_MS  # East
    v_d = np.zeros_like(v_n)                          # Down (не измеряется)
    gps_vel_ned = np.column_stack([v_n, v_e, v_d])

    # выравнивание длины vel и gnss (на случай различий)
    m = min(len(gps_pos_ned), len(gps_vel_ned))
    gps_pos_ned = gps_pos_ned[:m]
    gps_vel_ned = gps_vel_ned[:m]
    gps_time = gps_time[:m]
    gps_status = gps_status[:m]

    # -----//----- Индексы GPS в сетке IMU -----//-----
    gps_idx = np.searchsorted(imu_time, gps_time)
    gps_idx = np.clip(gps_idx, 0, len(imu_time) - 1)

    # -----//----- Инициализация из первой точки -----//-----
    init_r = np.zeros(3)                 # старт в локальном нуле NED
    init_v = gps_vel_ned[0].copy()       # первая измеренная скорость
    # Начальная ориентация из гравитации на стартовом покое (крен/тангаж),
    # вместо единичного кватерниона — иначе проекция g интегрируется в скорость.
    if align_init:
        n_align = max(1, int(align_sec * fs_imu))
        q0 = _gravity_alignment_q(accel, n_align)
    else:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])

    # -----//----- Шумы из конфигурации -----//-----
    noise = dict(
        sigma_a_n=IMU_ERR["accel_vrw"] / 60.0,
        sigma_g_n=np.deg2rad(IMU_ERR["gyro_arw"]) / 60.0,
        sigma_ba=IMU_ERR["accel_b_stability"],
        sigma_bg=np.deg2rad(IMU_ERR["gyro_b_stability"] / 3600.0),
        sigma_gps_pos=GPS_ERR["stdp"],
        sigma_gps_vel=GPS_ERR["stdv"],
    )

    S = {
        # IMU
        "accel": accel,
        "gyro": gyro,
        "imu_time": imu_time,
        # GPS (NED)
        "gps_pos_ned": gps_pos_ned,
        "gps_vel_ned": gps_vel_ned,
        "gps_idx": gps_idx,
        "gps_time": gps_time,
        "gps_status": gps_status,
        # инициализация
        "init_r": init_r,
        "init_v": init_v,
        "q0": q0,
        "ref_pos_ned": init_r[None, :],   # для совместимости с run_methods (берёт [0])
        "ref_vel_ned": init_v[None, :],
        "lat0_deg": lat0,
        "noise": noise,
        # метаданные опорной точки
        "ref_lla": (lat0, lon0, h0),
    }


    return S