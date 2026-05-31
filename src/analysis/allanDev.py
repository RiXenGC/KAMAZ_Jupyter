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
