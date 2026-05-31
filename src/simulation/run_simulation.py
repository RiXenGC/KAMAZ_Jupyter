from gnss_ins_sim.sim import imu_model, ins_sim
from src.config.config import IMU_ERR, GPS_ERR
from src.config.constants import FS_IMU, FS_GPS

mode = "0.8, 0.07, 0.1"
# mode = np.array([MAX_ACCEL_LAT, MAX_ACCEL_VERT, MAX_JERK])
# This is not implemented yet. A built-in 'high_mobility' mode is used.
#         or a numpy array of size (3,) to customize the sim mode.
#             [max_acceleration, max_angular_acceleration, max_angular_velocity],
#             in units of [m/s/s, deg/s/s, deg/s]


def run_simulation(motion_file, output_dir):
    """IMU Simulation"""
    fs = [FS_IMU, FS_GPS, 0.0]  # IMU, GPS, mag
    imu = imu_model.IMU(accuracy=IMU_ERR, axis=6, gps=True, gps_opt=GPS_ERR)

    sim = ins_sim.Sim(
        fs,
        motion_def=motion_file,
        ref_frame=0,  # 0 = NED, если конвертация в ENU — на этапе обработки
        imu=imu,
        mode=mode,
        env=None,
        #   env=env,
        algorithm=None,
    )

    sim.run(1)
    sim.results(output_dir)
    return sim
