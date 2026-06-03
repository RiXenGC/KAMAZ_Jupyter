import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize
from src.analysis.metrics import error_norm_series

# Import Times New Roman xDDD
import matplotlib.font_manager as fm

_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/times.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]

_loaded = False
for _fp in _FONT_CANDIDATES:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        try:
            plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
            _loaded = True
            break
        except Exception:
            pass

if not _loaded:
    plt.rcParams["font.family"] = "DejaVu Serif"


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 20,
        "axes.xmargin": 0,
        "legend.fontsize": 14,
        "legend.framealpha": 0.95,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

COLORS = {"x": "#E63946", "y": "#2A9D8F", "z": "#1D3557"}
COLOR_REF = "#264653"
COLOR_EST = "#E63946"


def _save(fig, save_dir, name):
    if save_dir is None:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_dir}/{name}.png", pad_inches=0.5)


# Графики ИИМ
def plot_accel_signal(time, accel, title_suffix="", save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, lbl in enumerate(["x", "y", "z"]):
        ax.plot(
            time,
            accel[:, i],
            color=COLORS[lbl],
            label=f"$a_{lbl}$",
            linewidth=0.7,
            alpha=0.9,
        )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Ускорение, м/с²")
    ax.set_title(f"Показания акселерометра в связанной СК{title_suffix}")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    _save(fig, save_dir, "accel_signal")
    return fig


def plot_gyro_signal(time, gyro, title_suffix="", save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    gyro_deg = np.rad2deg(gyro)
    for i, lbl in enumerate(["x", "y", "z"]):
        ax.plot(
            time,
            gyro_deg[:, i],
            color=COLORS[lbl],
            label=f"$\\omega_{lbl}$",
            linewidth=0.7,
            alpha=0.9,
        )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Угловая скорость, °/с")
    ax.set_title(f"Показания гироскопа в связанной СК{title_suffix}")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    _save(fig, save_dir, "gyro_signal")
    return fig


# Спектральный анализ
def plot_psd(
    time,
    signal_data,
    fs,
    channel_names=("x", "y", "z"),
    sensor_name="Акселерометр",
    units="м/с²",
    save_dir=None,
    fname="psd",
):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, lbl in enumerate(channel_names):
        f, Pxx = signal.welch(
            signal_data[:, i],
            fs=fs,
            nperseg=8192,
            noverlap=0.5 * 8192,
            nfft=16384,
            window="hamming",
            scaling="density",
        )
        ax.semilogy(
            f, Pxx, color=COLORS[lbl], label=f"Канал {lbl.upper()}", linewidth=1.3
        )

    # Вертикальные линии — ожидаемые гармоники ДВС
    for f_h, lbl in [(33.33, "1-я гарм."), (66.66, "2-я гарм."), (100.0, "3-я гарм.")]:
        ax.axvline(f_h, color="gray", linestyle=":", alpha=0.9, linewidth=1)
        ax.text(
            f_h,
            ax.get_ylim()[1] * 0.5,
            f" {lbl}\n {f_h:.1f} Гц",
            fontsize=12,
            color="gray",
            rotation=90,
            va="top",
        )

    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel(f"СПМ, ({units})²/Гц")
    ax.set_title(f"Спектральная плотность мощности — {sensor_name}")
    ax.set_xlim(0, fs / 2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, save_dir, fname)
    return fig


def plot_spectrogram_2d(
    time,
    signal_data_1ch,
    fs,
    channel_name="Z",
    sensor_name="Акселерометр",
    save_dir=None,
    fname="spectrogram_2d",
):
    fig, ax = plt.subplots(figsize=(10, 5))
    f, t_spec, Sxx = signal.spectrogram(
        signal_data_1ch,
        fs=fs,
        nperseg=2048,
        noverlap=1024,
        nfft=8192,
        window="hamming",
        scaling="density",
    )
    # дБ-шкала
    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    vmax = Sxx_db.max()
    vmin = vmax - 60

    im = ax.pcolormesh(
        t_spec, f, Sxx_db, shading="gouraud", cmap="turbo", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    ax.set_title(f"Спектрограмма — {sensor_name}, ось {channel_name}")
    ax.set_ylim(0, min(fs / 2, 200))  # сосредоточимся на диапазоне вибраций

    # Маркеры гармоник
    for f_h in [33.33, 66.66, 100.0]:
        ax.axhline(f_h, color="white", linestyle=":", alpha=0.9, linewidth=0.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Мощность, дБ")
    fig.tight_layout()
    _save(fig, save_dir, fname)
    return fig


def plot_spectrogram_3d(
    time,
    signal_data_1ch,
    fs,
    channel_name="Z",
    sensor_name="Акселерометр",
    save_dir=None,
    fname="spectrogram_3d",
):
    """Спектрограмма"""

    f, t_spec, Sxx = signal.spectrogram(
        signal_data_1ch,
        fs=fs,
        nperseg=2048,
        noverlap=1024,
        nfft=2048,
        window="hamming",
        scaling="density",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    # Ограничиваем частотный диапазон
    f_mask = f <= min(fs / 2, 200)
    f = f[f_mask]
    Sxx_db = Sxx_db[f_mask, :]

    # СРЕЗАЕМ ВЫБРОСЫ В МИНУС
    max_db = np.max(Sxx_db)
    min_db = max_db - 50.0
    Sxx_db_clipped = np.clip(Sxx_db, a_min=min_db, a_max=max_db)

    z_data = Sxx_db_clipped.T

    z_data_smooth = gaussian_filter(z_data, sigma=1.5)

    f_grid, t_grid = np.meshgrid(f, t_spec)

    # Инициализация 3D сцены
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # Отрисовка
    surf = ax.plot_surface(
        f_grid,
        t_grid,
        z_data_smooth,
        cmap="turbo",  # Классическая "инженерная" радужная палитра спектроанализаторов
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # Настройки осей и подписи
    ax.set_xlabel("Частота, Гц", labelpad=10)
    ax.set_ylabel("Время, с", labelpad=10)
    ax.set_zlabel("Мощность, дБ", labelpad=10)
    ax.set_title(
        f"Трёхмерная спектрограмма {sensor_name}, ось {channel_name}", fontsize=24
    )

    ax.view_init(elev=35, azim=-55)

    # Добавляем цветовую шкалу напрямую от поверхности `surf`
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label("Мощность, дБ")

    fig.tight_layout()
    _save(fig, save_dir, fname)
    return fig


# Девиация Аллана


def allan_deviation(data, fs, max_tau_factor=0.4):
    """Вычисление девиации Аллана методом overlapping samples"""
    N = len(data)
    t0 = 1.0 / fs
    # Логарифмическая сетка тау
    max_m = int(N * max_tau_factor)
    m_values = np.unique(np.logspace(0, np.log10(max_m), 100).astype(int))

    taus = m_values * t0
    sigmas = np.zeros(len(m_values))

    # Кумулятивная сумма для эффективности
    theta = np.cumsum(data) * t0

    for i, m in enumerate(m_values):
        # Overlapping Allan variance
        diff = theta[2 * m :] - 2 * theta[m:-m] + theta[: -2 * m]
        sigmas[i] = np.sqrt(np.mean(diff**2) / (2 * (m * t0) ** 2))

    return taus, sigmas


def plot_allan_simulated(gyro_static, accel_static, fs, save_dir=None):
    """Девиация Аллана для симулированных статических данных.
    Накладывать на референс из datasheet вы будете вручную."""
    # --- Гироскопы ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, lbl in enumerate(["x", "y", "z"]):
        gyro_deg_hr = np.rad2deg(gyro_static[:, i]) * 3600  # в °/ч
        taus, sigmas = allan_deviation(gyro_deg_hr, fs)
        ax.loglog(
            taus, sigmas, color=COLORS[lbl], label=f"Ось {lbl.upper()}", linewidth=2
        )

    ax.set_xlabel("Время осреднения τ, с")
    ax.set_ylabel("Отклонение Аллана, °/ч")
    ax.set_title("Диаграмма Аллана для гироскопов")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    _save(fig, save_dir, "allan_gyro")
    fig_gyro = fig

    # --- Акселерометры ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, lbl in enumerate(["x", "y", "z"]):
        acc_ug = accel_static[:, i] / 9.81 * 1e6  # в мкg
        taus, sigmas = allan_deviation(acc_ug, fs)
        ax.loglog(
            taus, sigmas, color=COLORS[lbl], label=f"Ось {lbl.upper()}", linewidth=2
        )

    ax.set_xlabel("Время осреднения τ, с")
    ax.set_ylabel("Отклонение Аллана, мкg")
    ax.set_title("Диаграмма Аллана для акселерометров")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    _save(fig, save_dir, "allan_accel")
    return fig_gyro, fig


# Reference trajectory
def plot_trajectory_3d(ref_pos_ned, est_pos_ned=None, save_dir=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        ref_pos_ned[:, 0],
        ref_pos_ned[:, 1],
        ref_pos_ned[:, 2],
        color=COLOR_REF,
        linewidth=2.2,
        label="Опорная траектория",
    )
    if est_pos_ned is not None:
        ax.plot(
            est_pos_ned[:, 0],
            est_pos_ned[:, 1],
            est_pos_ned[:, 2],
            color=COLOR_EST,
            linewidth=1.5,
            linestyle="--",
            label="Оценка БИНС",
        )
    ax.scatter(*ref_pos_ned[0], color="green", s=80, label="Старт", zorder=5)
    ax.scatter(*ref_pos_ned[-1], color="black", s=80, label="Финиш", zorder=5)
    ax.set_xlabel("North, м", labelpad=14)
    ax.set_ylabel("East, м", labelpad=14)
    ax.set_zlabel("Down, м", labelpad=14)
    ax.set_title("Траектория", pad=12)
    ax.legend(loc="upper left")
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, save_dir, "trajectory_3d")
    return fig


def plot_velocity_ned(time, ref_vel, est_vel=None, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = ["North", "East", "Down"]
    colors_v = [COLORS["x"], COLORS["y"], COLORS["z"]]
    for i, lbl in enumerate(labels):
        ax.plot(
            time,
            ref_vel[:, i],
            color=colors_v[i],
            linewidth=1.8,
            label=f"{lbl} (ист.)",
            zorder=10,
        )
        if est_vel is not None:
            ax.plot(
                time,
                est_vel[:, i],
                color=colors_v[i],
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
                label=f"{lbl} (БИНС)",
            )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Скорость, м/с")
    ax.set_title("Скорость")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    _save(fig, save_dir, "velocity_enu")
    return fig


def plot_euler_angles(time, ref_euler, est_euler=None, save_dir=None):
    figs = []
    for i, name in enumerate(["Рысканье", "Тангаж", "Крен"]):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(
            time,
            ref_euler[:, i],
            color=COLOR_REF,
            linewidth=1.8,
            label="Опорная",
            zorder=10,
        )
        if est_euler is not None:
            ax.plot(
                time,
                est_euler[:, i],
                color=COLOR_EST,
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                label="Оценка БИНС",
            )
        ax.set_xlabel("Время, с")
        ax.set_ylabel(f"{name}, °")
        ax.set_title(f"Угол ориентации: {name}")
        ax.legend(loc="best")
        fig.tight_layout()
        _save(fig, save_dir, f"euler_{name.lower()}")
        figs.append(fig)
    return figs


# ============ ОШИБКИ НАВИГАЦИИ ============
def plot_position_error(time, est_pos, ref_pos, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    err = est_pos - ref_pos
    err_norm = np.linalg.norm(err, axis=1)
    ax.plot(time, err[:, 0], label="Δ North", color=COLORS["x"], linewidth=1.1)
    ax.plot(time, err[:, 1], label="Δ East", color=COLORS["y"], linewidth=1.1)
    ax.plot(time, err[:, 2], label="Δ Down", color=COLORS["z"], linewidth=1.1)
    ax.plot(time, err_norm, label="|Δr|", color="black", linewidth=1.6, linestyle="--")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Ошибка положения, м")
    ax.set_title("Накопленная ошибка положения БИНС")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_dir, "position_error")
    return fig


def plot_attitude_error(time, est_euler, ref_euler, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    err = est_euler - ref_euler
    err[:, 0] = (err[:, 0] + 180) % 360 - 180  # wrap yaw
    ax.plot(time, err[:, 0], label="Δ Рысканье", color=COLORS["x"], linewidth=1.2)
    ax.plot(time, err[:, 1], label="Δ Тангаж", color=COLORS["y"], linewidth=1.2)
    ax.plot(time, err[:, 2], label="Δ Крен", color=COLORS["z"], linewidth=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Ошибка угла, °")
    ax.set_title("Ошибка оценки ориентации")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_dir, "attitude_error")
    return fig


# ============ ПАРАМЕТРИЧЕСКИЙ АНАЛИЗ БЕТТА ============
def plot_beta_sweep(time, results_dict, ref_euler, save_dir=None):
    figs = []
    cmap = plt.cm.viridis
    betas = sorted(results_dict.keys())
    colors_b = [cmap(i / max(len(betas) - 1, 1)) for i in range(len(betas))]

    for comp_idx, name in enumerate(["Рысканье", "Тангаж", "Крен"]):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            time,
            ref_euler[:, comp_idx],
            color="black",
            linewidth=2.0,
            label="Опорная",
            zorder=10,
        )
        count = len(betas) - 1
        for b, color in zip(betas, colors_b):
            est = results_dict[b]
            count -= 1
            ax.plot(
                time,
                est[:, comp_idx],
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=f"β = {b:.3f}",
                zorder=count,
            )
        ax.set_xlabel("Время, с")
        ax.set_ylabel(f"{name}, °")
        ax.set_title(f"Влияние параметра β фильтра Маджвика на оценку '{name}'")
        ax.legend(loc="best", ncol=2, fontsize=14)
        fig.tight_layout()
        _save(fig, save_dir, f"beta_sweep_{name.lower()}")
        figs.append(fig)

    # Сводный график RMSE по бета
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for comp_idx, name in enumerate(["Рысканье", "Тангаж", "Крен"]):
        rmse = []
        for b in betas:
            err = results_dict[b][:, comp_idx] - ref_euler[:, comp_idx]
            if comp_idx == 0:
                err = (err + 180) % 360 - 180
            rmse.append(np.sqrt(np.mean(err**2)))
        ax.plot(
            betas,
            rmse,
            marker="o",
            linewidth=1.5,
            color=COLORS[list(COLORS)[comp_idx]],
            label=name,
        )

    ax.set_xscale("log")
    ax.set_xlabel("β")
    ax.set_ylabel("RMSE угла, °")
    ax.set_title("Зависимость точности оценки ориентации от параметра β")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_dir, "beta_sweep_rmse")
    figs.append(fig)
    return figs


# Compare filters


def plot_overlay_trajectory(ref, ekf, ukf, fgo, save_dir=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(ref[:, 1], ref[:, 0], color="#264653", lw=2.5, label="Опорная")
    ax.plot(ekf[:, 1], ekf[:, 0], "--", color="#E63946", lw=1.6, label="EKF")
    ax.plot(ukf[:, 1], ukf[:, 0], "-.", color="#2A9D8F", lw=1.6, label="UKF")
    ax.plot(fgo[:, 1], fgo[:, 0], ":", color="#E76F51", lw=2.0, label="FGO")
    ax.scatter(ref[0, 1], ref[0, 0], c="green", s=80, zorder=5, label="Старт")
    ax.scatter(ref[-1, 1], ref[-1, 0], c="black", s=80, zorder=5, label="Финиш")
    ax.set_xlabel("East, м")
    ax.set_ylabel("North, м")
    ax.set_title("Траектория: сравнение методов")
    ax.legend()
    ax.axis("equal")
    ax.grid(alpha=0.3)
    _save(fig, save_dir, f"overlay_trajectory")
    return fig


def plot_overlay_error(t, ref, methods: dict, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"EKF": "#E63946", "UKF": "#2A9D8F", "FGO": "#E76F51"}
    for name, est in methods.items():
        ax.plot(
            t, error_norm_series(est, ref), lw=1.6, color=colors.get(name), label=name
        )
    ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("|Δr|, м (лог)")
    ax.set_title("Накопленная ошибка положения: EKF / UKF / FGO")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    _save(fig, save_dir, f"overlay_error")
    return fig
