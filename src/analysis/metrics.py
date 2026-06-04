import numpy as np


def _err(est, ref):
    est = np.asarray(est)
    ref = np.asarray(ref)
    if est.shape != ref.shape:
        raise ValueError(f"Формы не совпадают: est {est.shape} vs ref {ref.shape}")
    return est - ref


def rmse_vector(est, ref):
    """RMS модуля векторной ошибки (одно число)"""
    e = _err(est, ref)
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))


def rmse_per_axis(est, ref):
    """RMS по каждой оси отдельно → массив (3,)"""
    e = _err(est, ref)
    return np.sqrt(np.mean(e**2, axis=0))


def max_abs_error(est, ref):
    """Максимум модуля векторной ошибки"""
    e = _err(est, ref)
    return float(np.max(np.linalg.norm(e, axis=1)))


def final_error(est, ref):
    """Накопленная ошибка"""
    e = _err(est, ref)
    return float(np.linalg.norm(e[-1]))


def error_norm_series(est, ref):
    """|Δ|(t)"""
    e = _err(est, ref)
    return np.linalg.norm(e, axis=1)


def position_metrics(est_pos, ref_pos):
    axis = rmse_per_axis(est_pos, ref_pos)
    return {
        "rms_norm": rmse_vector(est_pos, ref_pos),
        "rms_north": float(axis[0]),
        "rms_east": float(axis[1]),
        "rms_down": float(axis[2]),
        "max": max_abs_error(est_pos, ref_pos),
        "final": final_error(est_pos, ref_pos),
    }


def velocity_metrics(est_vel, ref_vel):
    axis = rmse_per_axis(est_vel, ref_vel)
    return {
        "rms_norm": rmse_vector(est_vel, ref_vel),
        "rms_north": float(axis[0]),
        "rms_east": float(axis[1]),
        "rms_down": float(axis[2]),
        "max": max_abs_error(est_vel, ref_vel),
        "final": final_error(est_vel, ref_vel),
    }


def attitude_metrics(est_euler_deg, ref_euler_deg):
    """Ошибка приводится в диапазон [−180, 180]. Порядок столбцов: [yaw, pitch, roll]"""
    out = {}
    names = ['yaw', 'pitch', 'roll']
    for i, nm in enumerate(names):
        est_un = np.unwrap(np.deg2rad(est_euler_deg[:, i]))
        ref_un = np.unwrap(np.deg2rad(ref_euler_deg[:, i]))
        e = np.rad2deg(est_un - ref_un)
        out[f'rms_{nm}'] = float(np.sqrt(np.mean(e**2)))
        out[f'max_{nm}'] = float(np.max(np.abs(e)))
    return out


def compare_methods(results, ref_pos, ref_vel=None, ref_euler_deg=None):
    """
    results: dict {method: {'pos': est_pos, 'vel': est_vel, 'euler': est_euler_deg}}
    """
    rows = []
    for name, est in results.items():
        row = {"method": name}
        pm = position_metrics(est["pos"], ref_pos)
        row["pos_RMS_m"] = pm["rms_norm"]
        row["pos_max_m"] = pm["max"]
        row["pos_final_m"] = pm["final"]
        if ref_vel is not None and est.get("vel") is not None:
            row["vel_RMS_ms"] = velocity_metrics(est["vel"], ref_vel)["rms_norm"]
        if ref_euler_deg is not None and est.get("euler") is not None:
            am = attitude_metrics(est["euler"], ref_euler_deg)
            row["yaw_RMS_deg"] = am["rms_yaw"]
            row["pitch_RMS_deg"] = am["rms_pitch"]
            row["roll_RMS_deg"] = am["rms_roll"]
        rows.append(row)
    return rows


def print_comparison(results, ref_pos, ref_vel=None, ref_euler_deg=None):
    rows = compare_methods(results, ref_pos, ref_vel, ref_euler_deg)
    print(f"{'Метод':<12}{'RMS|dr|,м':>12}{'макс|dr|,м':>13}{'фин|dr|,м':>12}")
    print("-" * 49)
    for r in rows:
        print(
            f"{r['method']:<12}{r['pos_RMS_m']:>12.4f}"
            f"{r['pos_max_m']:>13.4f}{r['pos_final_m']:>12.4f}"
        )
    return rows
