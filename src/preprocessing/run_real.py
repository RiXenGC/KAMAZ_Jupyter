"""
run_real.py — прогон фильтров ИНС/СНС (EKF / UKF / FGO) на реальном сегменте
и сохранение отчётов (графики для одного метода и для сравнения трёх).
Ориентация берётся ТОЛЬКО из фильтра (измеренная недостоверна).
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

# -----//----- Пути проекта -----//-----
_THIS = Path(__file__).resolve()
for _p in (_THIS.parents[2], _THIS.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.preprocessing.real_data_adapter import prepare_real_segment

# единый стиль методов
try:
    from src.analysis.plot_style import METHOD_STYLE
except Exception:
    METHOD_STYLE = {
        "Измерено (СНС)": dict(color="k", lw=2.0, z=1),
        "EKF": dict(color="r", lw=1.5, z=3),
        "UKF": dict(color="g", lw=1.5, z=3),
        "FGO": dict(color="b", lw=1.5, z=3),
    }

METHODS = ("ekf", "ukf", "fgo")
REF_LABEL = "Измерено (СНС)"   # опорная = измеренный GNSS/VEL (не истина!)


# -----//----- Запуск одного фильтра -----//-----
def _run_one(method: str, S: dict):
    """Прогон одного фильтра. Возвращает (pos, vel, euler) в NED."""
    method = method.lower()
    if method == "ekf":
        from src.navigation.ekf import InsGnssEKF
        from src.analysis.run_methods import run_kalman
        return run_kalman(InsGnssEKF, S, S["accel"], S["gyro"])
    if method == "ukf":
        from src.navigation.ukf import InsGnssUKF
        from src.analysis.run_methods import run_kalman
        return run_kalman(InsGnssUKF, S, S["accel"], S["gyro"])
    if method == "fgo":
        from src.analysis.run_methods import run_fgo
        return run_fgo(S, S["accel"], S["gyro"])
    raise ValueError(f"method ∈ ekf/ukf/fgo, дано: {method}")


# -----//----- Сохранение -----//-----
def _save(fig, save_dir, name):
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_dir}/{name}.png", bbox_inches="tight", pad_inches=0.3)


def _gps_on_imu_grid(S):
    """Измеренные GPS позиция/скорость, разложенные на сетку IMU (для наложения)."""
    idx = S["gps_idx"]
    pos = np.full((len(S["imu_time"]), 3), np.nan)
    vel = np.full((len(S["imu_time"]), 3), np.nan)
    pos[idx] = S["gps_pos_ned"][:len(idx)]
    vel[idx] = S["gps_vel_ned"][:len(idx)]
    return pos, vel


# -----//----- Отчёт по одному методу -----//-----
def save_report_single(method: str, S, pos, vel, eul, out_dir):
    """Сохраняет графики одного метода: траектория, скорость, ориентация (из фильтра)."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t = S["imu_time"]
    gps_pos, gps_vel = _gps_on_imu_grid(S)
    s = METHOD_STYLE.get(method.upper(), dict(color="r", lw=1.5, z=3))

    # 1) траектория 3D (NED) + измеренный GPS
    fig = plt.figure(figsize=(9, 7.5)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=s["color"], lw=1.6,
            label=method.upper())
    ax.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], s=6,
               color="k", alpha=0.4, label=REF_LABEL)
    ax.scatter(*pos[0], c="#2A9D8F", s=90, label="Старт")
    ax.scatter(*pos[-1], c="#1D3557", s=90, marker="s", label="Финиш")
    ax.set_xlabel("North, м", labelpad=12); ax.set_ylabel("East, м", labelpad=12)
    ax.set_zlabel("Down, м", labelpad=8); ax.invert_zaxis(); ax.view_init(22, -60)
    ax.set_title(f"Траектория ({method.upper()})", fontweight="bold")
    ax.legend(loc="upper left")
    _save(fig, out_dir, "trajectory_3d"); plt.close(fig)

    # 2) траектория план (вид сверху)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(gps_pos[:, 1], gps_pos[:, 0], ".", ms=3, color="k", alpha=0.4,
            label=REF_LABEL)
    ax.plot(pos[:, 1], pos[:, 0], color=s["color"], lw=1.5, label=method.upper())
    ax.set_xlabel("East, м"); ax.set_ylabel("North, м")
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"Траектория, вид сверху ({method.upper()})", fontweight="bold")
    ax.legend()
    _save(fig, out_dir, "trajectory_NE"); plt.close(fig)

    # 3) скорость NED (оценка + измеренная)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, lbl, c in ((0, "North", "#E63946"), (1, "East", "#2A9D8F"), (2, "Down", "#1D3557")):
        ax.plot(t, vel[:, i], color=c, lw=1.5, label=f"{lbl} ({method.upper()})")
        ax.plot(t, gps_vel[:, i], ".", ms=2, color=c, alpha=0.35)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Скорость, м/с")
    ax.set_title(f"Скорость NED ({method.upper()})", fontweight="bold")
    ax.legend(ncol=3, fontsize=14)
    _save(fig, out_dir, "velocity_ned"); plt.close(fig)

    # 4) модуль скорости
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(t, np.linalg.norm(vel, axis=1), color=s["color"], lw=1.5,
            label=method.upper())
    gps_speed = np.linalg.norm(np.nan_to_num(gps_vel), axis=1)
    ax.plot(t, np.where(gps_speed > 0, gps_speed, np.nan), ".", ms=2,
            color="k", alpha=0.4, label=REF_LABEL)
    ax.set_xlabel("Время, с"); ax.set_ylabel("|v|, м/с")
    ax.set_title(f"Модуль скорости ({method.upper()})", fontweight="bold"); ax.legend()
    _save(fig, out_dir, "speed"); plt.close(fig)

    # 5) ориентация ИЗ ФИЛЬТРА (измеренная недостоверна — не выводим)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, i, name in zip(axes, range(3), ("Рысканье", "Тангаж", "Крен")):
        ax.plot(t, eul[:, i], color=s["color"], lw=1.5)
        ax.set_ylabel(f"{name}, °")
    axes[-1].set_xlabel("Время, с")
    fig.suptitle(f"Ориентация из фильтра ({method.upper()})", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "attitude_filter"); plt.close(fig)

    # данные оценки — npz, чтобы потом не пересчитывать
    np.savez(out_dir / "estimate.npz", t=t, pos=pos, vel=vel, euler=eul)

    # 6) ускорение IMU (то, что реально идёт в фильтр) — 3 оси
    fig, ax = plt.subplots(figsize=(11, 4.6))
    acc = S["accel"]
    for i, lbl, c in ((0, "a_x", "#E63946"), (1, "a_y", "#2A9D8F"), (2, "a_z", "#1D3557")):
        ax.plot(t, acc[:, i], color=c, lw=1.0, label=f"${lbl}$", alpha=0.9)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Ускорение, м/с²")
    ax.set_title(f"Ускорение IMU на входе фильтра ({method.upper()})", fontweight="bold")
    ax.legend(ncol=3)
    _save(fig, out_dir, "accel_input"); plt.close(fig)

    # 7) угловая скорость IMU — 3 оси
    fig, ax = plt.subplots(figsize=(11, 4.6))
    gyr = S["gyro"]
    for i, lbl, c in ((0, "\\omega_x", "#E63946"), (1, "\\omega_y", "#2A9D8F"),
                      (2, "\\omega_z", "#1D3557")):
        ax.plot(t, gyr[:, i], color=c, lw=1.0, label=f"${lbl}$", alpha=0.9)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Угловая скорость, рад/с")
    ax.set_title(f"Гироскоп на входе фильтра ({method.upper()})", fontweight="bold")
    ax.legend(ncol=3)
    _save(fig, out_dir, "gyro_input"); plt.close(fig)

    # 8) спектр ускорения — видно вибрацию ДВС и эффект ФНЧ
    try:
        from scipy import signal as _sig
        fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 400.0
        fig, ax = plt.subplots(figsize=(11, 4.6))
        for i, lbl, c in ((0, "X", "#E63946"), (1, "Y", "#2A9D8F"), (2, "Z", "#1D3557")):
            f, Pxx = _sig.welch(acc[:, i], fs=fs, nperseg=min(8192, len(t)),
                                window="hamming", scaling="density")
            ax.semilogy(f, Pxx, color=c, lw=1.5, label=f"Канал {lbl}")
        for fh, l in ((33.33, "1-я"), (66.66, "2-я"), (100.0, "3-я")):
            ax.axvline(fh, color="gray", ls=":", alpha=0.8, lw=1)
            ax.text(fh, ax.get_ylim()[1]*0.5, f" {l} ДВС\n {fh:.0f} Гц",
                    fontsize=12, color="gray", rotation=90, va="top")
        ax.set_xlabel("Частота, Гц"); ax.set_ylabel("СПМ, (м/с²)²/Гц")
        ax.set_title(f"Спектр ускорения ({method.upper()})", fontweight="bold")
        ax.set_xlim(0, fs/2); ax.legend()
        _save(fig, out_dir, "accel_psd"); plt.close(fig)
    except Exception as e:
        print(f"  (спектр пропущен: {e})")


# -----//----- Отчёт сравнения трёх методов -----//-----
def save_report_compare(S, results: dict, out_dir):
    """results: {'ekf': (p,v,e), 'ukf': (...), 'fgo': (...)}. Сохраняет наложения."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t = S["imu_time"]
    gps_pos, gps_vel = _gps_on_imu_grid(S)

    def _style(m):
        return METHOD_STYLE.get(m.upper(), dict(color="gray", lw=1.5, z=3))

    # 1) наложение траекторий (вид сверху)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(gps_pos[:, 1], gps_pos[:, 0], ".", ms=3, color="k", alpha=0.35,
            label=REF_LABEL)
    for m, (p, v, e) in results.items():
        st = _style(m)
        ax.plot(p[:, 1], p[:, 0], color=st["color"], lw=st["lw"], label=m.upper(),
                alpha=0.9)
    ax.set_xlabel("East, м"); ax.set_ylabel("North, м"); ax.set_aspect("equal", "datalim")
    ax.set_title("Сравнение траекторий (вид сверху)", fontweight="bold"); ax.legend()
    _save(fig, out_dir, "overlay_trajectory_NE"); plt.close(fig)

    # 2) наложение траекторий 3D
    fig = plt.figure(figsize=(9, 7.5)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2], s=5, color="k",
               alpha=0.3, label=REF_LABEL)
    for m, (p, v, e) in results.items():
        st = _style(m)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=st["color"], lw=st["lw"],
                label=m.upper(), alpha=0.9)
    ax.set_xlabel("North, м", labelpad=12); ax.set_ylabel("East, м", labelpad=12)
    ax.set_zlabel("Down, м", labelpad=8); ax.invert_zaxis(); ax.view_init(22, -60)
    ax.set_title("Сравнение траекторий 3D", fontweight="bold"); ax.legend(loc="upper left")
    _save(fig, out_dir, "overlay_trajectory_3d"); plt.close(fig)

    # 3) наложение модуля скорости
    fig, ax = plt.subplots(figsize=(11, 4.6))
    gps_speed = np.linalg.norm(np.nan_to_num(gps_vel), axis=1)
    ax.plot(t, np.where(gps_speed > 0, gps_speed, np.nan), ".", ms=2, color="k",
            alpha=0.4, label=REF_LABEL)
    for m, (p, v, e) in results.items():
        st = _style(m)
        ax.plot(t, np.linalg.norm(v, axis=1), color=st["color"], lw=st["lw"],
                label=m.upper(), alpha=0.9)
    ax.set_xlabel("Время, с"); ax.set_ylabel("|v|, м/с")
    ax.set_title("Сравнение модуля скорости", fontweight="bold")
    ax.legend(ncol=len(results) + 1)
    _save(fig, out_dir, "overlay_speed"); plt.close(fig)

    # 4) наложение углов из фильтров (по одному графику на угол)
    for i, which in enumerate(("Рысканье", "Тангаж", "Крен")):
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for m, (p, v, e) in results.items():
            st = _style(m)
            ax.plot(t, e[:, i], color=st["color"], lw=st["lw"], label=m.upper())
        ax.set_xlabel("Время, с"); ax.set_ylabel(f"{which}, °")
        ax.set_title(f"Сравнение: {which} (из фильтров)", fontweight="bold")
        ax.legend(ncol=len(results))
        _save(fig, out_dir, f"overlay_att_{which.lower()}"); plt.close(fig)

    # 5) попарное расхождение методов по позиции (нет эталона -> сравниваем между собой)
    if "ekf" in results and "ukf" in results and "fgo" in results:
        fig, ax = plt.subplots(figsize=(11, 4.6))
        pairs = [("ekf", "ukf"), ("ekf", "fgo"), ("ukf", "fgo")]
        for a, b in pairs:
            d = np.linalg.norm(results[a][0] - results[b][0], axis=1)
            ax.plot(t, d, lw=1.5, label=f"|{a.upper()}−{b.upper()}|")
        ax.set_xlabel("Время, с"); ax.set_ylabel("Расхождение позиции, м")
        ax.set_title("Попарное расхождение методов (нет эталона)", fontweight="bold")
        ax.legend()
        _save(fig, out_dir, "pairwise_divergence"); plt.close(fig)


# -----//----- Главная точка входа -----//-----
def run_segment(seg_dir, method: str = "ekf", out_dir=None,
                gnss_name="gnss0", vel_name="vel0", show=False) -> dict:
    """Прогон одного метода + сохранение отчёта.
    out_dir по умолчанию: <seg_dir>/reports/<method>/."""
    seg_dir = Path(seg_dir)
    S = prepare_real_segment(seg_dir, gnss_name=gnss_name, vel_name=vel_name)
    pos, vel, eul = _run_one(method, S)

    if out_dir is None:
        out_dir = seg_dir / "reports" / method.lower()
    save_report_single(method, S, pos, vel, eul, out_dir)

    print(f"[{method.upper()}] {pos.shape[0]} отсчётов | "
          f"RTK fix-доля: {(S['gps_status'] == 2).mean():.1%}")
    print(f"  отчёт сохранён: {out_dir}")
    if show:
        _show_quick(S, pos, vel, eul, method)
    return {"pos": pos, "vel": vel, "euler": eul, "S": S,
            "method": method, "out_dir": Path(out_dir)}


def run_all_methods(seg_dir, out_dir=None, gnss_name="gnss0", vel_name="vel0") -> dict:
    """Прогон всех трёх методов + отчёты по каждому + сравнительный отчёт.
    Структура: <seg_dir>/reports/{ekf,ukf,fgo,compare}/."""
    seg_dir = Path(seg_dir)
    S = prepare_real_segment(seg_dir, gnss_name=gnss_name, vel_name=vel_name)
    base = Path(out_dir) if out_dir is not None else seg_dir / "reports"

    results = {}
    for m in METHODS:
        pos, vel, eul = _run_one(m, S)
        results[m] = (pos, vel, eul)
        save_report_single(m, S, pos, vel, eul, base / m)
        print(f"[{m.upper()}] отчёт: {base / m}")

    save_report_compare(S, results, base / "compare")
    print(f"[COMPARE] сравнительный отчёт: {base / 'compare'}")
    print(f"  RTK fix-доля: {(S['gps_status'] == 2).mean():.1%}")
    return {"results": results, "S": S, "out_dir": base}


def _show_quick(S, pos, vel, eul, method):
    """Быстрый просмотр в ноутбуке (без сохранения)."""
    import matplotlib.pyplot as plt
    t = S["imu_time"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(S["gps_pos_ned"][:, 1], S["gps_pos_ned"][:, 0], ".", ms=3,
               color="k", alpha=0.4, label=REF_LABEL)
    ax[0].plot(pos[:, 1], pos[:, 0], color="r", lw=1.5, label=method.upper())
    ax[0].set_xlabel("East, м"); ax[0].set_ylabel("North, м")
    ax[0].set_aspect("equal", "datalim"); ax[0].legend(); ax[0].set_title("Траектория")
    ax[1].plot(t, np.linalg.norm(vel, axis=1), color="r", lw=1.5)
    ax[1].set_xlabel("Время, с"); ax[1].set_ylabel("|v|, м/с"); ax[1].set_title("Скорость")
    fig.tight_layout(); plt.show()


# -----//----- CLI -----//-----
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Прогон фильтра(ов) на реальном сегменте.")
    ap.add_argument("seg_dir")
    ap.add_argument("--method", default="ekf", choices=["ekf", "ukf", "fgo", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gnss", default="gnss0")
    ap.add_argument("--vel", default="vel0")
    a = ap.parse_args()
    if a.method == "all":
        run_all_methods(a.seg_dir, out_dir=a.out, gnss_name=a.gnss, vel_name=a.vel)
    else:
        run_segment(a.seg_dir, method=a.method, out_dir=a.out,
                    gnss_name=a.gnss, vel_name=a.vel)


if __name__ == "__main__":
    main()