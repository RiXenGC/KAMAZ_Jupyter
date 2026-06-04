import os
import math
import numpy as np
from gnss_ins_sim.sim import imu_model, ins_sim
from src.config.config import IMU_ERR
from src.config.constants import FS_IMU
from gnss_ins_sim.allan.allan import allan_var

motion = os.path.abspath(".//gnss_ins_sim//motion_def-Allan.csv")
fs = 100.0  # IMU sample frequency


def test_allan(output_dir):
    fs = [FS_IMU, 0.0, 0.0]  # IMU, GPS, mag
    imu = imu_model.IMU(accuracy=IMU_ERR, axis=6, gps=False)

    algo = allan_var

    #### start simulation
    sim = ins_sim.Sim(
        fs,
        motion,
        ref_frame=1,
        imu=imu,
        mode=None,
        env=None,
        algorithm=algo,
    )

    sim.run(1)
    sim.results(output_dir)  # save data files
    return sim


# if __name__ == "__main__":
#     test_allan()

# Девиация Аллана
# def allan_deviation(data, fs, max_tau_factor=0.4):
#     """Вычисление девиации Аллана методом overlapping samples"""
#     N = len(data)
#     t0 = 1.0 / fs
#     # Логарифмическая сетка тау
#     max_m = int(N * max_tau_factor)
#     m_values = np.unique(np.logspace(0, np.log10(max_m), 100).astype(int))

#     taus = m_values * t0
#     sigmas = np.zeros(len(m_values))

#     # Кумулятивная сумма для эффективности
#     theta = np.cumsum(data) * t0

#     for i, m in enumerate(m_values):
#         # Overlapping Allan variance
#         diff = theta[2 * m :] - 2 * theta[m:-m] + theta[: -2 * m]
#         sigmas[i] = np.sqrt(np.mean(diff**2) / (2 * (m * t0) ** 2))

#     return taus, sigmas


# def plot_allan_simulated(gyro_static, accel_static, fs, save_dir=None):
#     # --- Гироскопы
#     fig, ax = plt.subplots(figsize=(9, 5.5))
#     for i, lbl in enumerate(["x", "y", "z"]):
#         gyro_deg_hr = np.rad2deg(gyro_static[:, i]) * 3600  # в °/ч
#         taus, sigmas = allan_deviation(gyro_deg_hr, fs)
#         ax.loglog(
#             taus, sigmas, color=COLORS[lbl], label=f"Ось {lbl.upper()}", linewidth=2
#         )

#     ax.set_xlabel("Время осреднения τ, с")
#     ax.set_ylabel("Отклонение Аллана, °/ч")
#     ax.set_title("Диаграмма Аллана для гироскопов")
#     ax.legend(loc="best")
#     ax.grid(True, which="both", alpha=0.35)
#     fig.tight_layout()
#     _save(fig, save_dir, "allan_gyro")
#     fig_gyro = fig

#     # --- Акселерометры
#     fig, ax = plt.subplots(figsize=(9, 5.5))
#     for i, lbl in enumerate(["x", "y", "z"]):
#         acc_ug = accel_static[:, i] / 9.81 * 1e6  # в мкg
#         taus, sigmas = allan_deviation(acc_ug, fs)
#         ax.loglog(
#             taus, sigmas, color=COLORS[lbl], label=f"Ось {lbl.upper()}", linewidth=2
#         )

#     ax.set_xlabel("Время осреднения τ, с")
#     ax.set_ylabel("Отклонение Аллана, мкg")
#     ax.set_title("Диаграмма Аллана для акселерометров")
#     ax.legend(loc="best")
#     ax.grid(True, which="both", alpha=0.35)
#     fig.tight_layout()
#     _save(fig, save_dir, "allan_accel")
#     return fig_gyro, fig
