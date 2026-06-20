import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
from src.analysis.metrics import error_norm_series
from src.analysis.plot_style import METHOD_STYLE


def _style_axes(ax, logy=False):
    # рамка и сетка берутся из rcParams (plot_style) — как в visualization.py
    if not logy:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def _save(fig, save_dir, name):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{name}.png") 


# ---//--- Функции отрисовки ---//---


def plot_overlay_trajectory(ref, methods, plane='NE',
                            title=None, save_dir=None, name=None):
    """
    Наложение траекторий методов в выбранной плоскости проекции.

    plane:
      'NE' — вид сверху (North–East), горизонтальная плоскость;
      'ND' — вид сбоку (North–Down), вертикальный профиль вдоль North;
      'ED' — вид с торца (East–Down), вертикальный профиль вдоль East.
    methods: dict {'EKF': p_ekf, 'UKF': p_ukf, 'FGO': p_fgo}, массивы (N,3) в NED.
    """
    # индексы осей NED: 0=North, 1=East, 2=Down
    cfg = {
        'NE': dict(x=1, y=0, xl='East, м',  yl='North, м', eqax=True,  inv=False),
        'ND': dict(x=0, y=2, xl='North, м', yl='Down, м',  eqax=False, inv=True),
        'ED': dict(x=1, y=2, xl='East, м',  yl='Down, м',  eqax=False, inv=True),
    }
    if plane not in cfg:
        raise ValueError(f"plane должно быть 'NE', 'ND' или 'ED', получено {plane}")
    c = cfg[plane]
    titles = {
        'NE': 'Траектория (вид сверху, North–East)',
        'ND': 'Траектория (профиль, North–Down)',
        'ED': 'Траектория (профиль, East–Down)',
    }

    fig, ax = plt.subplots(figsize=(8.5, 7) if plane == 'NE' else (11, 4.6))

    st = METHOD_STYLE['Опорная']
    ax.plot(ref[:, c['x']], ref[:, c['y']], color=st['color'], lw=st['lw'],
            ls=st.get('ls', '-'), label='Опорная', zorder=st['z'])
    for name_m, p in methods.items():
        s = METHOD_STYLE.get(name_m, dict(color='gray', lw=1.5, ls='--', z=3))
        ax.plot(p[:, c['x']], p[:, c['y']], color=s['color'], lw=s['lw'],
                ls=s.get('ls', '-'), label=name_m, zorder=s['z'], alpha=0.9)

    ax.scatter(ref[0, c['x']], ref[0, c['y']], c='#2A9D8F', s=110, zorder=6,
               edgecolors='white', linewidths=1.5, label='Старт')
    ax.scatter(ref[-1, c['x']], ref[-1, c['y']], c='#1D3557', s=110, zorder=6,
               marker='s', edgecolors='white', linewidths=1.5, label='Финиш')

    ax.set_xlabel(c['xl']); ax.set_ylabel(c['yl'])
    ax.set_title(title or titles[plane], fontweight='bold', pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=11, loc='best')
    if c['eqax']:
        ax.set_aspect('equal', adjustable='datalim')
    if c['inv']:
        ax.invert_yaxis()          # Down растёт вниз — естественно для профиля
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, save_dir, name or f"overlay_trajectory_{plane}")
    return fig


def plot_overlay_trajectory_3d(
    ref,
    methods,
    title="Траектория движения",
    save_dir=None,
    name="overlay_trajectory_3d",
):
    """3D-траектория (North, East, Down) с наложением методов"""
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    st = METHOD_STYLE["Опорная"]
    ax.plot(
        ref[:, 0],
        ref[:, 1],
        ref[:, 2],
        color=st["color"],
        lw=st["lw"],
        label="Опорная",
        zorder=st["z"],
    )
    for name_m, p in methods.items():
        s = METHOD_STYLE.get(name_m, dict(color="gray", lw=1.6, z=3))
        ax.plot(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            color=s["color"],
            lw=s["lw"],
            label=name_m,
            alpha=0.9,
        )
    ax.scatter(
        *ref[0], c="#2A9D8F", s=90, edgecolors="white", linewidths=1.5, label="Старт"
    )
    ax.scatter(
        *ref[-1],
        c="#1D3557",
        s=90,
        marker="s",
        edgecolors="white",
        linewidths=1.5,
        label="Финиш",
    )
    ax.set_xlabel("North, м", labelpad=14)
    ax.set_ylabel("East, м", labelpad=14)
    ax.set_zlabel("Down, м", labelpad=8)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    ax.invert_zaxis()

    # видимость подписи Down
    ax.view_init(elev=22, azim=-60)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    fig.subplots_adjust(right=0.85)        # освободить место справа под Down

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{name}.png", bbox_inches="tight", pad_inches=0.3)
    return fig


def plot_overlay_speed(
    t,
    ref_vel,
    methods_vel,
    title="Модуль линейной скорости",
    save_dir=None,
    name="overlay_speed",
):
    """
    Сравнение модуля скорости |v| трёх методов с эталоном.
    ref_vel, methods_vel[...] — массивы (N,3) в NED, м/с.
    """
    fig, ax = plt.subplots(figsize=(11, 4.6))
    st = METHOD_STYLE["Опорная"]
    ax.plot(
        t,
        np.linalg.norm(ref_vel, axis=1),
        color=st["color"],
        lw=st["lw"],
        label="Опорная",
        zorder=st["z"],
    )
    for name_m, v in methods_vel.items():
        s = METHOD_STYLE.get(name_m, dict(color="gray", lw=1.6, z=3))
        ax.plot(
            t,
            np.linalg.norm(v, axis=1),
            color=s["color"],
            lw=s["lw"],
            label=name_m,
            zorder=s["z"],
            alpha=0.9,
        )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("|v|, м/с")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        ncol=len(methods_vel) + 1,
    )
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, save_dir, name)
    return fig


def plot_overlay_attitude(
    t, ref_euler_deg, methods, which="yaw", save_dir=None, name=None
):
    """
    Ошибка одного конкретного угла ориентации
    """
    idx = {"yaw": 0, "pitch": 1, "roll": 2}[which]
    titles = {
        "yaw": "Ошибка курса",
        "pitch": "Ошибка тангажа",
        "roll": "Ошибка крена",
    }

    ref_un = np.rad2deg(np.unwrap(np.deg2rad(ref_euler_deg[:, idx])))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for name_m, eul in methods.items():
        s = METHOD_STYLE.get(name_m, dict(color="gray", lw=1.6, z=3))

        est_un = np.rad2deg(np.unwrap(np.deg2rad(eul[:, idx])))
        err = est_un - ref_un

        ax.plot(
            t,
            err,
            color=s["color"],
            lw=s["lw"],
            label=name_m,
            zorder=s["z"],
        )
    ax.axhline(0, color="#1D3557", lw=1.0, alpha=0.5)
    ax.set_xlabel("Время, с")
    ax.set_ylabel(f"{titles[which]}, °")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=len(methods))
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, save_dir, name or f"overlay_att_{which}")
    return fig


def plot_overlay_yaw_rate(
    t,
    ref_euler_deg,
    methods_euler,
    fs=400.0,
    ref_yaw_rate=None,
    title="Угловая скорость (рысканье)",
    save_dir=None,
    name="overlay_yawrate",
):
    """
    Угловая скорость рысканья (°/с) трёх методов.
    Эталон: ref_yaw_rate (если задан, рад/с из ref_gyro) либо производная
    эталонного рысканья. Оценки методов получаются дифференцированием их
    рысканья по времени (сглаженное).
    """

    def yaw_rate_from_euler(eul_deg):
        yaw = np.deg2rad(eul_deg[:, 0])
        yaw = np.unwrap(yaw)  # снять скачки ±180°
        rate = np.gradient(yaw, 1.0 / fs)  # рад/с
        # лёгкое сглаживание (диф. шумит)
        win = max(1, int(fs * 0.1))  # окно 0.1 с
        if win > 1:
            kern = np.ones(win) / win
            rate = np.convolve(rate, kern, mode="same")
        return np.rad2deg(rate)  # °/с

    fig, ax = plt.subplots(figsize=(11, 4.6))
    # эталон
    st = METHOD_STYLE["Опорная"]
    if ref_yaw_rate is not None:
        ref_r = np.rad2deg(ref_yaw_rate[:, 2])  # z-компонента gyro = yaw rate
    else:
        ref_r = yaw_rate_from_euler(ref_euler_deg)
    ax.plot(
        t,
        ref_r,
        color=st["color"],
        lw=st["lw"],
        label="Опорная",
        zorder=st["z"],
    )
    for name_m, eul in methods_euler.items():
        s = METHOD_STYLE.get(name_m, dict(color="gray", lw=1.6, z=3))
        ax.plot(
            t,
            yaw_rate_from_euler(eul),
            color=s["color"],
            lw=s["lw"],
            label=name_m,
            zorder=s["z"],
            alpha=0.9,
        )
    ax.set_xlabel("Время, с")
    ax.set_ylabel("ω_yaw, °/с")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        ncol=len(methods_euler) + 1,
    )
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, save_dir, name)
    return fig


# ---//--- Ошибки ---//---


def plot_overlay_error(
    t,
    ref,
    methods,
    title="Накопленная ошибка положения",
    logy=True,
    save_dir=None,
    name="overlay_error",
):
    """
    Ошибка положения
    """
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for name_m, p in methods.items():
        s = METHOD_STYLE.get(name_m, dict(color="gray", lw=1.6, z=3))
        e = error_norm_series(p, ref)
        ax.plot(t, e, color=s["color"], lw=s["lw"], label=name_m, zorder=s["z"])
        ax.fill_between(t, e, alpha=0.07, color=s["color"])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("|Δr|, м" + (" (лог)" if logy else ""))
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=len(methods))
    _style_axes(ax, logy=logy)
    fig.tight_layout()
    _save(fig, save_dir, name)
    return fig


def plot_overlay_velocity_error(
    t,
    ref_vel,
    methods_vel,
    title="Ошибка определения скорости",
    logy=True,
    save_dir=None,
    name="overlay_velocity_error",
):
    """Модуль ошибки скорости |Δv|(t) для всех методов."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for name_m, v in methods_vel.items():
        s = METHOD_STYLE.get(name_m, dict(color="gray", lw=1.6, z=3))
        e = np.linalg.norm(v - ref_vel, axis=1)
        ax.plot(t, e, color=s["color"], lw=s["lw"], label=name_m, zorder=s["z"])
        ax.fill_between(t, e, alpha=0.07, color=s["color"])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("|Δv|, м/с" + (" (лог)" if logy else ""))
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        ncol=len(methods_vel),
    )
    _style_axes(ax, logy=logy)
    fig.tight_layout()
    _save(fig, save_dir, name)
    return fig


def plot_velocity_error_axes(
    t, ref_vel, est_vel, method_name="EKF", save_dir=None, name=None
):
    """Ошибка скорости по осям N/E/D для ОДНОГО метода."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    axes_lbl = ["North", "East", "Down"]
    axes_col = ["#E63946", "#2A9D8F", "#1D3557"]
    e = est_vel - ref_vel
    for i in range(3):
        ax.plot(t, e[:, i], color=axes_col[i], lw=1.6, label=f"Δv {axes_lbl[i]}")
    ax.axhline(0, color="#1D3557", lw=1.0, alpha=0.5)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Δv, м/с")
    ax.set_title(f"Ошибка скорости по осям ({method_name})", fontweight="bold", pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=3)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, save_dir, name or f"velocity_error_axes_{method_name}")
    return fig


# Сравнение чистого и с вибрациями


def plot_clean_vs_vib(
    t, ref, res_clean, res_vib, method="EKF", save_dir=None, name=None
):
    """
    Влияние вибрации: |Δr|(t) одного метода — чисто vs с вибрацией.
    res_clean, res_vib: dict из run_all_methods (берётся [method]['pos']).
    """
    fig, ax = plt.subplots(figsize=(11, 4.6))
    e_clean = error_norm_series(res_clean[method]["pos"], ref)
    e_vib = error_norm_series(res_vib[method]["pos"], ref)
    ax.plot(t, e_clean, color="#2A9D8F", lw=1.9, label=f"{method}: без вибрации")
    ax.fill_between(t, e_clean, alpha=0.08, color="#2A9D8F")
    ax.plot(t, e_vib, color="#E63946", lw=1.9, ls="--", label=f"{method}: с вибрацией")
    ax.fill_between(t, e_vib, alpha=0.08, color="#E63946")
    ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("|Δr|, м (лог)")
    ax.set_title(
        f"Влияние вибрации силовой установки ({method})", fontweight="bold", pad=12
    )
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")
    _style_axes(ax, logy=True)
    fig.tight_layout()
    _save(fig, save_dir, name or f"clean_vs_vib_{method}")
    return fig
