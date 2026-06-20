"""
segmentation.py — инструмент ручной нарезки логов на сценарии.

Категории: stop_EngOff, stop_EngOn, gruz, dvij.
"""

# from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

CATEGORIES = ("stop_EngOff", "stop_EngOn", "gruz", "dvij")
CAT_COLORS = {
    "stop_EngOff": "#9e9e9e",   # серый — стоянка без двигателя
    "stop_EngOn":  "#ffb300",   # жёлтый — стоянка с двигателем
    "gruz":        "#8e24aa",   # фиолетовый — погрузка
    "dvij":        "#2e7d32",   # зелёный — движение
}

IMU_FS = 400.0   # Гц, для окна RMS
GNSS_FS = 10.0

@dataclass
class Seg:
    """Один размеченный сегмент: [start, end] сек, категория."""
    start: float
    end: float
    label: str

    def __post_init__(self):
        if self.label not in CATEGORIES:
            raise ValueError(f"label '{self.label}' не из {CATEGORIES}")
        if self.end <= self.start:
            raise ValueError(f"end({self.end}) <= start({self.start})")


# -----//----- Загрузка -----//-----
def load_processed(proc_dir) -> dict:
    """Грузит все parquet из папки обработки в dict DataFrame."""
    
    proc_dir = Path(proc_dir)
    
    data = {}
    for nm in ("gnss0", "gnss1", "vel0", "vel1", "imu"):
        p = proc_dir / f"{nm}.parquet"
        if p.exists():
            data[nm] = pd.read_parquet(p)
            
    meta_p = proc_dir / "window_meta.parquet"
    if meta_p.exists():
        data["meta"] = pd.read_parquet(meta_p)
    # сводка
    for nm in ("gnss0", "vel0", "imu"):
        if nm in data:
            d = data[nm]
            print(f"[{nm}] {d.shape}, t: {d['t'].min():.1f} -> {d['t'].max():.1f} c")
    return data


# -----//----- ПОДГОТОВКА СИГНАЛОВ ДЛЯ ОБЗОРА -----//-----
def _rolling_rms(x: np.ndarray, win: int) -> np.ndarray:
    """RMS в скользящем окне (через свёртку квадрата)."""
    win = max(1, int(win))
    x2 = np.asarray(x, dtype=np.float64) ** 2
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(x2, kernel, mode="same"))


def _decimate_to(t: np.ndarray, y: np.ndarray, n_target: int):
    """Прореживание длинного ряда для отрисовки: берём максимум по бинам
    (огибающая сохраняется, всплески не теряются)."""
    n = len(t)
    if n <= n_target:
        return t, y
    step = int(np.ceil(n / n_target))
    n_bins = n // step
    t_b = t[:n_bins * step].reshape(n_bins, step)[:, 0]
    y_b = y[:n_bins * step].reshape(n_bins, step).max(axis=1)
    return t_b, y_b


def build_overview_signals(data: dict, rms_win_sec: float = 1.0,
                            n_plot: int = 8000) -> dict:
    """Готовит сигналы для обзорного графика. IMU децимируется до n_plot точек.

    Возвращает dict с (t, y) для каждой панели:
      speed   — модуль скорости ГНСС, км/ч
      acc_rms — RMS горизонтального ускорения IMU (вибрации), м/с^2
      gyro_rms— RMS угловой скорости IMU, рад/с
      altitude— высота ГНСС, м
    """
    out = {}

    # скорость
    if "vel0" in data:
        v = data["vel0"]
        out["speed"] = (v["t"].to_numpy(), v["speed_kmh"].to_numpy())

    # высота
    if "gnss0" in data:
        g = data["gnss0"]
        out["altitude"] = (g["t"].to_numpy(), g["altitude"].to_numpy())

    # IMU RMS (на полной сетке, потом децимация для рисования)
    if "imu" in data:
        imu = data["imu"]
        t = imu["t"].to_numpy()
        win = int(rms_win_sec * IMU_FS)
        # горизонтальные ускорения (после поворота гравитация в Z, X/Y — динамика)
        ax = imu["linear_acceleration_x"].to_numpy()
        ay = imu["linear_acceleration_y"].to_numpy()
        acc_h = np.sqrt(ax**2 + ay**2)
        acc_rms = _rolling_rms(acc_h, win)
        gyro_mag = np.sqrt(imu["angular_velocity_x"].to_numpy()**2
                           + imu["angular_velocity_y"].to_numpy()**2
                           + imu["angular_velocity_z"].to_numpy()**2)
        gyro_rms = _rolling_rms(gyro_mag, win)
        out["acc_rms"] = _decimate_to(t, acc_rms, n_plot)
        out["gyro_rms"] = _decimate_to(t, gyro_rms, n_plot)

    return out


# ======================================================================
#                          ОБЗОРНЫЙ ГРАФИК
# ======================================================================
def overview(data: dict, segments: list | None = None,
             rms_win_sec: float = 1.0, figsize=(15, 9)):
    """Обзорный график всей записи: 4 панели с общей осью времени.
    Если переданы segments — подсвечивает их цветом категории."""
    import matplotlib.pyplot as plt

    sig = build_overview_signals(data, rms_win_sec=rms_win_sec)
    panels = [
        ("speed",    "Скорость ГНСС, км/ч",      "#1565c0"),
        ("acc_rms",  "RMS ускор. (вибр.), м/с²", "#c62828"),
        ("gyro_rms", "RMS гироскопа, рад/с",     "#6a1b9a"),
        ("altitude", "Высота, м",                "#00838f"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    for ax, (key, ylab, col) in zip(axes, panels):
        if key in sig:
            t, y = sig[key]
            ax.plot(t, y, color=col, lw=0.7)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, alpha=0.3)
        if segments:
            for s in segments:
                ax.axvspan(s.start, s.end, color=CAT_COLORS[s.label], alpha=0.25)
    axes[-1].set_xlabel("Время, с (от начала окна)")
    if segments:
        from matplotlib.patches import Patch
        handles = [Patch(color=CAT_COLORS[c], alpha=0.4, label=c) for c in CATEGORIES]
        axes[0].legend(handles=handles, ncol=4, loc="upper right", fontsize=8)
    fig.suptitle("Обзор записи — навигация для разметки", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ======================================================================
#                          ДЕТАЛЬНЫЙ ЗУМ
# ======================================================================
def zoom(data: dict, t_start: float, t_end: float,
         show_raw_imu: bool = True, figsize=(15, 10)):
    """Детальный график интервала [t_start, t_end]. IMU здесь рисуется сырым
    (на коротком окне точек немного), чтобы точно поставить границы."""
    import matplotlib.pyplot as plt

    def _slice(df):
        m = (df["t"] >= t_start) & (df["t"] <= t_end)
        return df[m]

    rows = []
    if "vel0" in data:
        rows.append("speed")
    if "imu" in data:
        rows += ["acc_raw", "gyro_raw"] if show_raw_imu else ["acc_rms", "gyro_rms"]
    if "gnss0" in data:
        rows.append("altitude")

    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, sharex=True)
    if len(rows) == 1:
        axes = [axes]
    ai = 0

    if "vel0" in data:
        v = _slice(data["vel0"])
        axes[ai].plot(v["t"], v["speed_kmh"], color="#1565c0", lw=1.0)
        axes[ai].set_ylabel("Скорость, км/ч"); axes[ai].grid(alpha=0.3); ai += 1

    if "imu" in data:
        imu = _slice(data["imu"])
        t = imu["t"].to_numpy()
        if show_raw_imu:
            axes[ai].plot(t, imu["linear_acceleration_x"], lw=0.5, label="a_x")
            axes[ai].plot(t, imu["linear_acceleration_y"], lw=0.5, label="a_y")
            axes[ai].plot(t, imu["linear_acceleration_z"], lw=0.5, label="a_z")
            axes[ai].set_ylabel("Ускор., м/с²"); axes[ai].legend(fontsize=7, ncol=3)
            axes[ai].grid(alpha=0.3); ai += 1
            axes[ai].plot(t, imu["angular_velocity_x"], lw=0.5, label="ω_x")
            axes[ai].plot(t, imu["angular_velocity_y"], lw=0.5, label="ω_y")
            axes[ai].plot(t, imu["angular_velocity_z"], lw=0.5, label="ω_z")
            axes[ai].set_ylabel("Угл.скор., рад/с"); axes[ai].legend(fontsize=7, ncol=3)
            axes[ai].grid(alpha=0.3); ai += 1

    if "gnss0" in data:
        g = _slice(data["gnss0"])
        axes[ai].plot(g["t"], g["altitude"], color="#00838f", lw=1.0)
        axes[ai].set_ylabel("Высота, м"); axes[ai].grid(alpha=0.3); ai += 1

    axes[-1].set_xlabel("Время, с")
    fig.suptitle(f"Зум [{t_start} … {t_end}] с — точная разметка границ", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ======================================================================
#                       НАРЕЗКА И СОХРАНЕНИЕ
# ======================================================================
def cut_segment(data: dict, seg: Seg, zero_time: bool = True) -> dict:
    """Вырезает все потоки по интервалу seg. zero_time=True сдвигает время к 0
    (как nac в Matlab: SAT.Time = SAT.Time - nac)."""
    res = {}
    for nm in ("gnss0", "gnss1", "vel0", "vel1", "imu"):
        if nm not in data:
            continue
        df = data[nm]
        m = (df["t"] >= seg.start) & (df["t"] <= seg.end)
        sub = df[m].copy().reset_index(drop=True)
        if zero_time and len(sub):
            sub["t"] = sub["t"] - seg.start
        res[nm] = sub
    return res


def cut_and_save(data: dict, segments: list, out_dir, zero_time: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
    """Режет все сегменты и сохраняет в out_dir/<label>/<label>_<i>/<sensor>.parquet.
    Возвращает таблицу-манифест размеченных сегментов."""
    out_dir = Path(out_dir)
    counters = {c: 0 for c in CATEGORIES}
    rows = []
    for seg in segments:
        counters[seg.label] += 1
        idx = counters[seg.label]
        seg_dir = out_dir / seg.label / f"{seg.label}_{idx}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        cut = cut_segment(data, seg, zero_time=zero_time)
        for nm, sub in cut.items():
            sub.to_parquet(seg_dir / f"{nm}.parquet", index=False)
        n_imu = len(cut.get("imu", []))
        n_gnss = len(cut.get("gnss0", []))
        rows.append({
            "label": seg.label, "index": idx,
            "start_s": seg.start, "end_s": seg.end,
            "duration_s": seg.end - seg.start,
            "n_gnss": n_gnss, "n_imu": n_imu, "dir": str(seg_dir),
        })
        if verbose:
            print(f"[{seg.label}_{idx}] {seg.start:.1f}–{seg.end:.1f} c "
                  f"({seg.end-seg.start:.1f} c), gnss={n_gnss}, imu={n_imu}")
    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_dir / "segments_manifest.parquet", index=False)
    manifest.to_csv(out_dir / "segments_manifest.csv", index=False)
    if verbose:
        print(f"\n[OK] {len(segments)} сегментов сохранено в {out_dir}")
        print(manifest.groupby('label')['duration_s'].agg(['count', 'sum']))
    return manifest


def validate_segments(segments: list, total_dur: float | None = None) -> None:
    """Проверка списка сегментов: пересечения, выход за границы записи."""
    ss = sorted(segments, key=lambda s: s.start)
    for a, b in zip(ss, ss[1:]):
        if b.start < a.end:
            print(f"! ПЕРЕСЕЧЕНИЕ: {a.label}({a.start}-{a.end}) и "
                  f"{b.label}({b.start}-{b.end})")
    if total_dur is not None:
        for s in ss:
            if s.end > total_dur:
                print(f"! ВЫХОД ЗА ЗАПИСЬ: {s.label} end={s.end} > {total_dur:.1f}")
    print(f"Проверено сегментов: {len(segments)}")
