"""
segmentation.py — ручная нарезка логов на сценарии: загрузка parquet,
графики (обзор / зум / 3D-траектория), нарезка и сохранение сегментов.
Категории: stop_EngOff, stop_EngOn, gruz, dvij.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# единый стиль графиков (Times New Roman, мягкая сетка, xmargin=0, font 20)
try:
    from src.analysis import plot_style  # noqa: F401  — rcParams при импорте
except Exception:
    try:
        import plot_style  # noqa: F401  — локальный фолбэк
    except Exception:
        pass

# -----//----- Константы (из src.config.constants) -----//-----
try:
    from src.config.constants import (
        CATEGORIES, CAT_COLORS, COL_SPEED, COL_AX, COL_AY, COL_AZ,
        FS_IMU as IMU_FS, FS_GNSS as GNSS_FS, OR_POINT, A_WGS84,
    )
except Exception:  # запуск вне пакета
    from constants import (
        CATEGORIES, CAT_COLORS, COL_SPEED, COL_AX, COL_AY, COL_AZ,
        FS_IMU as IMU_FS, FS_GNSS as GNSS_FS, OR_POINT, A_WGS84,
    )


def latlon_to_local(lat, lon, h, ref=OR_POINT):
    """Перевод (широта, долгота, высота) -> локальные метры ENU от опорной точки.
    Эквидистантная проекция: точна на масштабах карьера (километры).
    Возвращает (xE, yN, zUp) — восток, север, вверх, в метрах."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    lat0, lon0, h0 = ref
    lat0_rad = np.deg2rad(lat0)
    xE = np.deg2rad(lon - lon0) * A_WGS84 * np.cos(lat0_rad)
    yN = np.deg2rad(lat - lat0) * A_WGS84
    zUp = h - h0
    return xE, yN, zUp


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


# -----//----- Подготовка сигналов обзора -----//-----
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


def build_overview_signals(data: dict, n_plot: int = 8000) -> dict:
    """Сигналы для обзорного графика. IMU децимируется до n_plot точек
    с сохранением огибающей (min и max по бинам — шум виден честно).

    Возвращает dict:
      speed — модуль скорости ГНСС, м/с (t, y)
      a_x, a_y, a_z — сырое линейное ускорение IMU по трём осям, м/с²
                      каждое как (t, y_min, y_max) для отрисовки полосы шума
    """
    out = {}
    if "vel0" in data:
        v = data["vel0"]
        out["speed"] = (v["t"].to_numpy(), v["speed_kmh"].to_numpy() / 3.6)  # м/с

    if "imu" in data:
        imu = data["imu"]
        t = imu["t"].to_numpy()
        for key, col in (("a_x", "linear_acceleration_x"),
                         ("a_y", "linear_acceleration_y"),
                         ("a_z", "linear_acceleration_z")):
            y = imu[col].to_numpy()
            out[key] = _decimate_envelope(t, y, n_plot)
    return out


def _decimate_envelope(t, y, n_target):
    """Прореживание с сохранением огибающей: для каждого бина — min и max,
    чтобы зашумлённый сигнал на обзоре выглядел как полоса, а не терял пики."""
    n = len(t)
    if n <= n_target:
        return t, y, y
    step = int(np.ceil(n / n_target))
    nb = n // step
    tb = t[:nb * step].reshape(nb, step)[:, 0]
    yb = y[:nb * step].reshape(nb, step)
    return tb, yb.min(axis=1), yb.max(axis=1)


# -----//----- Обзорный график -----//-----
def overview(data: dict, segments: list | None = None, figsize=(15, 10)):
    """Обзорный график всей записи: скорость ГНСС + сырое линейное ускорение
    по трём осям (зашумлённое, как полоса огибающей). Общая ось времени.
    Если переданы segments — подсвечивает их цветом категории."""
    import matplotlib.pyplot as plt

    sig = build_overview_signals(data)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # 1. скорость
    if "speed" in sig:
        t, y = sig["speed"]
        axes[0].plot(t, y, color=COL_SPEED, lw=1.5)
    axes[0].set_ylabel("V$_{ГНСС}$, м/с")

    # 2-4. ускорения как полоса огибающей (min..max) + средняя линия
    for ax, key, col, lab in (
        (axes[1], "a_x", COL_AX, "a$_x$, м/с²"),
        (axes[2], "a_y", COL_AY, "a$_y$, м/с²"),
        (axes[3], "a_z", COL_AZ, "a$_z$, м/с²"),
    ):
        if key in sig:
            t, ymin, ymax = sig[key]
            ax.fill_between(t, ymin, ymax, color=col, alpha=0.55, lw=0)
            ax.plot(t, (ymin + ymax) / 2, color=col, lw=1.5, alpha=0.9)
        ax.set_ylabel(lab)

    for ax in axes:
        if segments:
            for s in segments:
                ax.axvspan(s.start, s.end, color=CAT_COLORS[s.label], alpha=0.45, lw=0)

    axes[-1].set_xlabel("Время, с")
    if segments:
        from matplotlib.patches import Patch
        handles = [Patch(color=CAT_COLORS[c], alpha=0.6, label=c) for c in CATEGORIES]
        axes[0].legend(handles=handles, ncol=4, loc="upper right")
    fig.suptitle("Обзор записи", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# -----//----- Детальный зум -----//-----
def zoom(data: dict, t_start: float, t_end: float,
         show_raw_imu: bool = True, figsize=(15, 11)):
    """Детальный график интервала [t_start, t_end]. IMU рисуется сырым.
    Внизу — панель наличия RTK-поправки (status==2) для обоих приёмников."""
    import matplotlib.pyplot as plt

    def _slice(df):
        m = (df["t"] >= t_start) & (df["t"] <= t_end)
        return df[m]

    rows = []
    if "vel0" in data:
        rows.append("speed")
    if "imu" in data:
        rows += ["acc", "gyro"]
    if "gnss0" in data or "gnss1" in data:
        rows.append("rtk")

    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, sharex=True)
    if len(rows) == 1:
        axes = [axes]
    ai = 0

    if "vel0" in data:
        v = _slice(data["vel0"])
        axes[ai].plot(v["t"], v["speed_kmh"] / 3.6, color=COL_SPEED, lw=1.5)
        axes[ai].set_ylabel("V, м/с"); ai += 1

    if "imu" in data:
        imu = _slice(data["imu"])
        t = imu["t"].to_numpy()
        axes[ai].plot(t, imu["linear_acceleration_x"], lw=0.6, color=COL_AX, label="a$_x$")
        axes[ai].plot(t, imu["linear_acceleration_y"], lw=0.6, color=COL_AY, label="a$_y$")
        axes[ai].plot(t, imu["linear_acceleration_z"], lw=0.6, color=COL_AZ, label="a$_z$")
        axes[ai].set_ylabel("a, м/с²"); axes[ai].legend(ncol=3, fontsize=13); ai += 1
        axes[ai].plot(t, imu["angular_velocity_x"], lw=0.6, color=COL_AX, label="ω$_x$")
        axes[ai].plot(t, imu["angular_velocity_y"], lw=0.6, color=COL_AY, label="ω$_y$")
        axes[ai].plot(t, imu["angular_velocity_z"], lw=0.6, color=COL_AZ, label="ω$_z$")
        axes[ai].set_ylabel("ω, рад/с"); axes[ai].legend(ncol=3, fontsize=13); ai += 1

    # --- RTK-панель: полоса наличия поправки (status==2) ---
    if "gnss0" in data or "gnss1" in data:
        ax = axes[ai]
        levels = []  # (y-уровень, имя, df)
        if "gnss0" in data:
            levels.append((1.0, "ГНСС0", _slice(data["gnss0"])))
        if "gnss1" in data:
            levels.append((0.0, "ГНСС1", _slice(data["gnss1"])))
        yticks, ylabels = [], []
        for y, nm, g in levels:
            tt = g["t"].to_numpy()
            fix = (g["status"].to_numpy() == 2)
            # зелёная полоса там, где есть RTK fix, серая фоном где нет
            ax.fill_between(tt, y - 0.32, y + 0.32, where=fix,
                            color="#66bb6a", alpha=0.85, step="mid", lw=0)
            ax.fill_between(tt, y - 0.32, y + 0.32, where=~fix,
                            color="#e0e0e0", alpha=0.7, step="mid", lw=0)
            yticks.append(y); ylabels.append(nm)
        ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
        ax.set_ylim(-0.6, 1.6)
        ax.set_ylabel("RTK")
        ax.grid(False)
        # легенда наличия/отсутствия
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#66bb6a", alpha=0.85, label="RTK=1"),
                           Patch(color="#e0e0e0", alpha=0.7, label="RTK=0")],
                  ncol=2, fontsize=13, loc="upper right")
        ai += 1

    axes[-1].set_xlabel("Время, с")
    fig.suptitle(f"Зум [{t_start:g} … {t_end:g}] с", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# -----//----- Графики сегмента (траектория 3D + скорость) -----//-----
def save_segment_figures(seg_data: dict, out_dir, title_suffix=""):
    """Сохраняет 3D-траекторию (NED, стиль сравнения, без опорной точки) и
    скорость (м/с) для одного среза данных seg_data в out_dir."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 3D-траектория (NED, приближённая, без опорной точки) ---
    if "gnss0" in seg_data and len(seg_data["gnss0"]):
        g = seg_data["gnss0"]
        xE, yN, zUp = latlon_to_local(
            g["latitude"].to_numpy(), g["longitude"].to_numpy(),
            g["altitude"].to_numpy())
        north, east, down = yN, xE, -zUp   # ENU -> NED
        fix = g["status"].to_numpy() == 2

        fig = plt.figure(figsize=(9, 7.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(north, east, down, color="#90a4ae", lw=1.5, alpha=0.7, zorder=1)
        ax.scatter(north[fix], east[fix], down[fix], s=5, c="#2A9D8F",
                   label="RTK=1", depthshade=True)
        ax.scatter(north[~fix], east[~fix], down[~fix], s=5, c="#cfd8dc",
                   label="RTK=0", depthshade=True)
        ax.scatter(north[0], east[0], down[0], s=90, c="#2A9D8F", marker="o",
                   edgecolors="k", label="Старт")
        ax.scatter(north[-1], east[-1], down[-1], s=90, c="#1D3557", marker="s",
                   edgecolors="k", label="Финиш")
        ax.set_xlabel("North, м", labelpad=12)
        ax.set_ylabel("East, м", labelpad=12)
        ax.set_zlabel("Down, м", labelpad=8)
        ax.invert_zaxis(); ax.view_init(22, -60)
        ax.set_title(f"Траектория (СНС){title_suffix}", fontweight="bold")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_3d.png", bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

    # --- скорость (м/с) ---
    if "vel0" in seg_data and len(seg_data["vel0"]):
        v = seg_data["vel0"]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(v["t"], v["speed_kmh"] / 3.6, color=COL_SPEED, lw=1.5)
        ax.set_xlabel("Время, с"); ax.set_ylabel("|v|, м/с")
        ax.set_title(f"Модуль скорости (СНС){title_suffix}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "speed.png", bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


# -----//----- Нарезка и сохранение -----//-----
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
                 save_figures: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Режет все сегменты и сохраняет в out_dir/<label>/<label>_<i>/<sensor>.parquet.
    При save_figures=True для каждого сегмента сохраняет trajectory_3d.png и
    speed.png в его папку. Возвращает таблицу-манифест."""
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
        # графики именно этого нарезанного сегмента
        if save_figures:
            try:
                save_segment_figures(cut, seg_dir,
                                     title_suffix=f" — {seg.label}_{idx}")
            except Exception as e:
                if verbose:
                    print(f"  (графики {seg.label}_{idx} пропущены: {e})")
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


# -----//----- FigureTabs — окно с вкладками -----//-----
class FigureTabs:
    """Интерфейс разметки в одном окне с вкладками (внутри ноутбука).

    Вкладки:
      • Обзор      — 4 панели на всю запись (навигация)
      • Зум        — детальный интервал по полям ввода (точные границы)
      • Траектория — путь по ГНСС с RTK-подсветкой

    Использование:
        import segmentation as sg
        data = sg.load_processed(PROC_DIR)
        ui = sg.FigureTabs(data)
        ui.show()
        # размечаешь, заполняешь список Seg, режешь cut_and_save(...)

    Не требует Qt/DISPLAY — рисует в выводе ячейки. Для зума по мыши нужен
    ipympl (%matplotlib widget), но основной режим зума — по числовым полям.
    """

    def __init__(self, data: dict):
        self.data = data
        self.t_max = self._compute_tmax()

    def _compute_tmax(self) -> float:
        tm = 0.0
        for nm in ("imu", "vel0", "gnss0"):
            if nm in self.data:
                tm = max(tm, float(self.data[nm]["t"].max()))
        return tm

    # -----//----- Вкладка: обзор -----//-----
    def _render_overview(self, out, segments=None):
        import matplotlib.pyplot as plt
        with out:
            out.clear_output(wait=True)
            fig, axes = overview(self.data, segments=segments)
            plt.show()

    # -----//----- Вкладка: зум -----//-----
    def _render_zoom(self, out, t0, t1, raw=True):
        import matplotlib.pyplot as plt
        with out:
            out.clear_output(wait=True)
            if t1 <= t0:
                print("Конец должен быть больше начала.")
                return
            fig, axes = zoom(self.data, t0, t1, show_raw_imu=raw)
            plt.show()

    # -----//----- Вкладка: 3D-траектория -----//-----
    def _render_traj(self, out, t0=None, t1=None):
        import matplotlib.pyplot as plt
        with out:
            out.clear_output(wait=True)
            if "gnss0" not in self.data:
                print("Нет gnss0.")
                return
            g = self.data["gnss0"]
            if t0 is not None and t1 is not None:
                g = g[(g["t"] >= t0) & (g["t"] <= t1)]
            if not len(g):
                print("Нет точек в интервале.")
                return

            # перевод в локальные метры ENU от опорной точки GlobalVars
            xE, yN, zUp = latlon_to_local(
                g["latitude"].to_numpy(), g["longitude"].to_numpy(),
                g["altitude"].to_numpy())
            fix = g["status"].to_numpy() == 2

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111, projection="3d")
            # линия пути
            ax.plot(xE, yN, zUp, color="#90a4ae", lw=0.8, alpha=0.7, zorder=1)
            # точки по RTK-статусу
            ax.scatter(xE[fix], yN[fix], zUp[fix], s=4, c="#66bb6a",
                       label="RTK fix", depthshade=True)
            ax.scatter(xE[~fix], yN[~fix], zUp[~fix], s=4, c="#cfd8dc",
                       label="нет fix", depthshade=True)
            # опорная точка, старт, финиш
            ax.scatter([0], [0], [0], s=90, c="k", marker="o",
                       label="O (опорная точка)")
            ax.scatter([xE[0]], [yN[0]], [zUp[0]], s=110, c="#e53935",
                       marker="o", edgecolors="k", label="старт")
            ax.scatter([xE[-1]], [yN[-1]], [zUp[-1]], s=110, c="#1e88e5",
                       marker="o", edgecolors="k", label="финиш")

            ax.set_xlabel("X (восток), м", labelpad=12)
            ax.set_ylabel("Y (север), м", labelpad=12)
            ax.set_zlabel("Z (верх), м", labelpad=8)
            ax.set_title("Траектория в локальной СК (RTK-статус цветом)")
            ax.legend(loc="upper left", fontsize=13)
            # равный масштаб по X и Y (горизонталь не искажается)
            try:
                ax.set_box_aspect((1, 1, 0.4))
            except Exception:
                pass
            fig.tight_layout()
            plt.show()

    # -----//----- Сборка UI -----//-----
    def show(self, segments=None):
        import ipywidgets as W
        from IPython.display import display

        out_ov = W.Output()
        out_zm = W.Output()
        out_tr = W.Output()

        # --- управление зумом ---
        f_start = W.FloatText(value=0.0, description="старт, с",
                              layout=W.Layout(width="180px"))
        f_end = W.FloatText(value=min(300.0, self.t_max), description="конец, с",
                            layout=W.Layout(width="180px"))
        raw_cb = W.Checkbox(value=True, description="сырой IMU")
        btn_zoom = W.Button(description="Показать зум", button_style="primary")
        zoom_ctrl = W.HBox([f_start, f_end, raw_cb, btn_zoom])

        def _on_zoom(_):
            self._render_zoom(out_zm, f_start.value, f_end.value, raw_cb.value)
        btn_zoom.on_click(_on_zoom)

        # --- управление траекторией (опц. интервал) ---
        t_start = W.FloatText(value=0.0, description="старт, с",
                              layout=W.Layout(width="180px"))
        t_end = W.FloatText(value=self.t_max, description="конец, с",
                            layout=W.Layout(width="180px"))
        btn_tr = W.Button(description="Показать траекторию", button_style="primary")
        tr_ctrl = W.HBox([t_start, t_end, btn_tr])

        def _on_tr(_):
            self._render_traj(out_tr, t_start.value, t_end.value)
        btn_tr.on_click(_on_tr)

        tab_ov = W.VBox([out_ov])
        tab_zm = W.VBox([zoom_ctrl, out_zm])
        tab_tr = W.VBox([tr_ctrl, out_tr])

        tabs = W.Tab(children=[tab_ov, tab_zm, tab_tr])
        tabs.set_title(0, "Обзор")
        tabs.set_title(1, "Зум")
        tabs.set_title(2, "Траектория")

        # первичная отрисовка
        self._render_overview(out_ov, segments=segments)
        self._render_zoom(out_zm, f_start.value, f_end.value, raw_cb.value)
        self._render_traj(out_tr)

        display(tabs)
        self._tabs = tabs
        self._out_ov = out_ov
        return tabs

    def refresh_overview(self, segments):
        """Перерисовать обзор с подсветкой размеченных сегментов."""
        self._render_overview(self._out_ov, segments=segments)