"""
preprocess.py — предобработка и синхронизация 4 потоков датчиков
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

NS = 1_000_000_000
G = 9.80665

LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)
ALT_RANGE = (-500.0, 5000.0)
VEL_ABS_MAX_KMH = 150.0
ACC_NORM_RANGE = (3.0, 40.0)
GYRO_ABS_MAX = 10.0
MAD_K = 8.0


def _sec(ns):
    return ns.astype(np.float64) / NS


def _sort_unique_time(df, tcol="t"):
    df = df.sort_values(tcol, kind="mergesort")
    df = df[~df[tcol].duplicated(keep="first")]
    return df.reset_index(drop=True)


def _mad_mask(x, k=MAD_K):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return np.isfinite(x)
    return np.abs(x - med) <= k * 1.4826 * mad


def _interp_grid(t_src, vals, t_grid):
    out = np.empty((t_grid.size, vals.shape[1]), dtype=np.float64)
    for j in range(vals.shape[1]):
        out[:, j] = np.interp(t_grid, t_src, vals[:, j])
    return out


@dataclass
class Window:
    t0_ns: int
    start_ns: int
    end_ns: int
    warmup_sec: float = 0.0

    @property
    def duration_s(self):
        return (self.end_ns - self.start_ns) / NS

    @property
    def lo(self):
        return (self.start_ns - self.t0_ns) / NS

    @property
    def hi(self):
        return (self.end_ns - self.t0_ns) / NS


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_mounting_matrix(tilt_x_deg=0.0, tilt_y_deg=0.0, tilt_z_deg=0.0):
    """Матрица выравнивания датчик -> объект (NED body).
    Компенсирует наклон установки поворотом на ОТРИЦАТЕЛЬНЫЕ углы.
    Порядок (обратный установке): сначала -Oz (доворот), затем -Oy, затем -Ox.

    ВАЖНО: для этих данных 16.5° — крен вокруг Ox (не Oy!): гравитация -2.80
    сидит в оси Y, а g*sin(16.5)=2.786 это подтверждает. Поворот вокруг Ox
    ставит её в +Z. Углы оставлены раздельными по всем трём осям, чтобы можно
    было задать любую конвенцию."""
    ax = np.deg2rad(tilt_x_deg)
    ay = np.deg2rad(tilt_y_deg)
    az = np.deg2rad(tilt_z_deg)
    return _rot_x(-ax) @ _rot_y(-ay) @ _rot_z(-az)


def apply_imu_mounting(imu, tilt_x_deg=0.0, tilt_y_deg=0.0, tilt_z_deg=0.0, verbose=True):
    """Поворачивает accel и gyro IMU из системы датчика в систему объекта (NED).
    Контроль: гравитация покоя после поворота должна лечь в +Z (~+9.81)."""
    R = build_mounting_matrix(tilt_x_deg, tilt_y_deg, tilt_z_deg)
    acc = imu[["linear_acceleration_x", "linear_acceleration_y",
               "linear_acceleration_z"]].to_numpy()
    gyr = imu[["angular_velocity_x", "angular_velocity_y",
               "angular_velocity_z"]].to_numpy()
    acc_b = acc @ R.T
    gyr_b = gyr @ R.T
    out = imu.copy()
    out[["linear_acceleration_x", "linear_acceleration_y",
         "linear_acceleration_z"]] = acc_b
    out[["angular_velocity_x", "angular_velocity_y",
         "angular_velocity_z"]] = gyr_b
    if verbose:
        pre = acc.mean(axis=0)
        post = acc_b.mean(axis=0)
        print(f"[mounting] R для Ox={tilt_x_deg}°, Oy={tilt_y_deg}°, Oz={tilt_z_deg}°")
        print(f"[mounting] mean accel ДО : [{pre[0]:+.3f}, {pre[1]:+.3f}, {pre[2]:+.3f}], |g|={np.linalg.norm(pre):.3f}")
        print(f"[mounting] mean accel ПОСЛЕ: [{post[0]:+.3f}, {post[1]:+.3f}, {post[2]:+.3f}], |g|={np.linalg.norm(post):.3f}")
        if post[2] < 0:
            print("[mounting] ! гравитация в -Z: инвертируй знак угла наклона.")
        elif post[2] < 0.9 * np.linalg.norm(post):
            print("[mounting] ! гравитация не легла в +Z — проверь знаки углов.")
        else:
            print("[mounting] OK: гравитация преимущественно в +Z (NED).")
    return out


def _last_timestamp_csv(path, tcol_index=0, tail_bytes=65536):
    """Читает последний timestamp в большом CSV без чтения всего файла:
    seek к концу, забираем хвост, парсим первый столбец последней полной строки."""
    import os
    path = str(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        seek_pos = max(0, size - tail_bytes)
        f.seek(seek_pos)
        tail = f.read().decode("utf-8", errors="ignore")
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Не удалось прочитать хвост {path}")
    # последняя строка может быть обрезана сверху — берём именно последнюю целую
    last_line = lines[-1]
    # защита: если хвост начался с середины строки, последняя строка всё равно целая
    return int(last_line.split(",")[tcol_index])


def compute_window(path_g0, path_g1, path_imu, path_vel, warmup_sec, imu_chunk, verbose=True):
    starts, ends = [], []
    for p in (path_g0, path_g1, path_vel):
        t = pd.read_csv(p, usecols=["timestamp"])["timestamp"].to_numpy()
        starts.append(int(t.min()))
        ends.append(int(t.max()))
    # IMU: первый timestamp дёшево (1 строка), последний — через seek в конец файла,
    # не читая весь файл целиком (timestamp — первый столбец, индекс 0).
    first = pd.read_csv(path_imu, usecols=["timestamp"], nrows=1)["timestamp"].iloc[0]
    last = _last_timestamp_csv(path_imu, tcol_index=0)
    starts.append(int(first))
    ends.append(int(last))
    t0_ns = max(starts)
    end_ns = min(ends)
    start_ns = t0_ns + int(warmup_sec * NS)
    if start_ns >= end_ns:
        raise ValueError("Окно после warmup пустое — проверь warmup_sec и данные.")
    w = Window(t0_ns, start_ns, end_ns, warmup_sec=float(warmup_sec))
    if verbose:
        print(f"[window] t0={t0_ns}  start(+{warmup_sec:g}s)={start_ns}  end={end_ns}")
        print(f"[window] валидное окно: {w.duration_s:.1f} c ({w.duration_s/60:.1f} мин)")
    return w


def process_gnss(path, w, name, fs, verbose=True):
    raw = pd.read_csv(path)
    raw["t"] = _sec(raw["timestamp"].to_numpy()) - w.t0_ns / NS
    raw = raw[raw["status"] != -1]
    raw = raw.dropna(subset=["longitude", "latitude", "altitude"])
    raw = _sort_unique_time(raw, "t")
    raw = raw[(raw["t"] >= w.lo) & (raw["t"] <= w.hi)].reset_index(drop=True)
    m = (raw["latitude"].between(*LAT_RANGE) & raw["longitude"].between(*LON_RANGE)
         & raw["altitude"].between(*ALT_RANGE))
    raw = raw[m].reset_index(drop=True)
    keep = _mad_mask(raw["latitude"].to_numpy()) & _mad_mask(raw["longitude"].to_numpy())
    raw = raw[keep].reset_index(drop=True)
    t_grid = np.arange(0.0, w.hi - w.lo + 1e-9, 1.0 / fs)
    t_src = raw["t"].to_numpy() - w.lo
    coords = raw[["longitude", "latitude", "altitude"]].to_numpy()
    g = _interp_grid(t_src, coords, t_grid)
    status_grid = np.rint(np.interp(t_grid, t_src, raw["status"].to_numpy())).astype(np.int8)
    out = pd.DataFrame({"t": t_grid, "longitude": g[:, 0], "latitude": g[:, 1],
                        "altitude": g[:, 2], "status": status_grid})
    if verbose:
        print(f"[{name}] валидных: {len(raw)}, на сетке: {len(out)}")
    return out


def process_vel(path, w, fs, verbose=True):
    raw = pd.read_csv(path, usecols=["timestamp", "linear_velocity_x",
                                     "linear_velocity_y", "linear_velocity_z"])
    raw["t_ns"] = raw["timestamp"].to_numpy()
    raw = raw.sort_values("t_ns", kind="mergesort")
    raw = raw[~raw["t_ns"].duplicated(keep="first")].reset_index(drop=True)
    dt_ns = np.diff(raw["t_ns"].to_numpy(), prepend=raw["t_ns"].iloc[0])
    PAIR_GAP_NS = 30_000_000
    recv = (dt_ns < PAIR_GAP_NS).astype(int)
    recv[0] = 0
    raw["recv"] = recv
    vel0 = raw[raw["recv"] == 0].copy()
    vel1 = raw[raw["recv"] == 1].copy()
    if verbose:
        print(f"[VEL] vel0={len(vel0)}, vel1={len(vel1)}, дисбаланс={abs(len(vel0)-len(vel1))}")
    t_grid = np.arange(0.0, w.hi - w.lo + 1e-9, 1.0 / fs)

    def _one(df, tag):
        df = df.copy()
        df["t"] = _sec(df["t_ns"].to_numpy()) - w.t0_ns / NS
        df = df[(df["t"] >= w.lo) & (df["t"] <= w.hi)].reset_index(drop=True)
        vx = df["linear_velocity_x"].to_numpy() * 3.6
        vy = df["linear_velocity_y"].to_numpy() * 3.6
        speed = np.sqrt(vx**2 + vy**2)
        keep = (speed <= VEL_ABS_MAX_KMH) & _mad_mask(speed)
        df, vx, vy, speed = df[keep], vx[keep], vy[keep], speed[keep]
        t_src = df["t"].to_numpy() - w.lo
        g = _interp_grid(t_src, np.column_stack([vx, vy, speed]), t_grid)
        out = pd.DataFrame({"t": t_grid, "v_x_kmh": g[:, 0],
                            "v_y_kmh": g[:, 1], "speed_kmh": g[:, 2]})
        if verbose:
            print(f"[VEL{tag}] валидных: {len(df)}, на сетке: {len(out)}")
        return out

    return _one(vel0, "0"), _one(vel1, "1")


def process_imu(path, w, imu_chunk, tilt_x_deg=None, tilt_y_deg=None, tilt_z_deg=None, verbose=True):
    cols = ["timestamp", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
    lo_ns, hi_ns = w.start_ns, w.end_ns
    parts, total_raw, total_kept = [], 0, 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=imu_chunk):
        total_raw += len(ch)
        t_ns = ch["timestamp"].to_numpy()
        in_win = (t_ns >= lo_ns) & (t_ns <= hi_ns)
        if not in_win.any():
            continue
        ch = ch[in_win]
        acc = ch[["linear_acceleration_x", "linear_acceleration_y",
                  "linear_acceleration_z"]].to_numpy()
        gyro = ch[["angular_velocity_x", "angular_velocity_y",
                   "angular_velocity_z"]].to_numpy()
        acc_norm = np.linalg.norm(acc, axis=1)
        m_acc = (acc_norm >= ACC_NORM_RANGE[0]) & (acc_norm <= ACC_NORM_RANGE[1])
        m_gyro = np.all(np.abs(gyro) <= GYRO_ABS_MAX, axis=1)
        m_fin = np.all(np.isfinite(acc), axis=1) & np.all(np.isfinite(gyro), axis=1)
        ch = ch[m_acc & m_gyro & m_fin]
        total_kept += len(ch)
        if len(ch):
            parts.append(ch)
    imu = pd.concat(parts, ignore_index=True)
    # Единый ноль времени = start (после warmup), как у GNSS/VEL.
    # start_ns = t0_ns + warmup, поэтому вычитаем start_ns/NS = t0/NS + lo.
    imu["t"] = _sec(imu["timestamp"].to_numpy()) - w.start_ns / NS
    imu = _sort_unique_time(imu, "t")
    dt = np.diff(imu["t"].to_numpy(), prepend=imu["t"].iloc[0])
    imu["gap_before"] = dt > (3.0 / 400.0)
    out = imu[["t", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
               "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
               "gap_before"]].reset_index(drop=True)
    if verbose:
        print(f"[IMU] прочитано: {total_raw}, в окне+валидных: {total_kept}, "
              f"итог: {len(out)}, разрывов: {int(out['gap_before'].sum())}, "
              f"~{1/np.median(np.diff(out['t'])):.1f} Гц")
    if tilt_x_deg is not None:
        out = apply_imu_mounting(out, tilt_x_deg, tilt_y_deg or 0.0,
                                 tilt_z_deg or 0.0, verbose=verbose)
    return out


def run(rec_dir, out_dir, warmup_sec=30.0, fs_gnss=10.0, imu_chunk=2_000_000,
        imu_tilt_deg=(16.5, 0.0, -0.5), save=True, verbose=True):
    """Полный прогон. Возвращает dict с DataFrame (gnss0/gnss1/vel0/vel1/imu/meta/window)."""
    rec_dir, out_dir = Path(rec_dir), Path(out_dir)
    p_g0, p_g1 = rec_dir / "gnss0_data.csv", rec_dir / "gnss1_data.csv"
    p_imu, p_vel = rec_dir / "imu_data.csv", rec_dir / "vel_data.csv"
    if verbose:
        print("=== 1. Окно ===")
    w = compute_window(p_g0, p_g1, p_imu, p_vel, warmup_sec, imu_chunk, verbose)
    if verbose:
        print("\n=== 2. GNSS ===")
    g0 = process_gnss(p_g0, w, "GNSS0", fs_gnss, verbose)
    g1 = process_gnss(p_g1, w, "GNSS1", fs_gnss, verbose)
    if verbose:
        print("\n=== 3. VEL ===")
    v0, v1 = process_vel(p_vel, w, fs_gnss, verbose)
    if verbose:
        print("\n=== 4. IMU ===")
    tilt = imu_tilt_deg if imu_tilt_deg is not None else (None, None, None)
    imu = process_imu(p_imu, w, imu_chunk, tilt[0], tilt[1], tilt[2], verbose)
    meta = pd.DataFrame([{
        "t0_ns": w.t0_ns, "start_ns": w.start_ns, "end_ns": w.end_ns,
        "warmup_sec": warmup_sec, "duration_s": w.duration_s, "fs_gnss": fs_gnss,
        "imu_tilt_x_deg": tilt[0], "imu_tilt_y_deg": tilt[1],
        "imu_tilt_z_deg": tilt[2]}])
    result = {"gnss0": g0, "gnss1": g1, "vel0": v0, "vel1": v1,
              "imu": imu, "meta": meta, "window": w}
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        g0.to_parquet(out_dir / "gnss0.parquet", index=False)
        g1.to_parquet(out_dir / "gnss1.parquet", index=False)
        v0.to_parquet(out_dir / "vel0.parquet", index=False)
        v1.to_parquet(out_dir / "vel1.parquet", index=False)
        imu.to_parquet(out_dir / "imu.parquet", index=False)
        meta.to_parquet(out_dir / "window_meta.parquet", index=False)
        if verbose:
            print(f"\n[OK] сохранено в {out_dir}")
    return result
