import os
from src.config.constants import INIT_LAT, INIT_LON, INIT_ALT

HEADER = (
    "ini lat (deg),ini lon (deg),ini alt (m),"
    "ini vx_body (m/s),ini vy_body (m/s),ini vz_body (m/s),"
    "ini yaw (deg),ini pitch (deg),ini roll (deg)\n"
    "{lat},{lon},{alt},{vx},{vy},{vz},{yaw},{pitch},{roll}\n"
    "command type,yaw (deg),pitch (deg),roll (deg),"
    "vx_body (m/s),vy_body (m/s),vz_body (m/s),"
    "command duration (s),GPS visibility\n"
)

INIT_DEFAULT = {
    "lat": INIT_LAT,
    "lon": INIT_LON,
    "alt": INIT_ALT,
    "vx": 0.0,
    "vy": 0.0,
    "vz": 0.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
}

INIT_CRUISE = {
    "lat": INIT_LAT,
    "lon": INIT_LON,
    "alt": INIT_ALT,
    "vx": 11.11,
    "vy": 0.0,
    "vz": 0.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
}

INIT_TURN = {
    "lat": INIT_LAT,
    "lon": INIT_LON,
    "alt": INIT_ALT,
    "vx": 5.56,
    "vy": 0.0,
    "vz": 0.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
}


def write_profile(path, init_state, commands):
    """
    init_state: dict с lat, lon, alt, yaw, pitch, roll
    commands: список строк с командами
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER.format(**init_state))
        for cmd in commands:
            f.write(cmd + "\n")


def gen_stationary(path):
    """Сценарий 1: Стоянка (30 с)"""
    cmds = ["1,0,0,0,0,0,0,30,1"]  # удержание состояния
    write_profile(path, INIT_DEFAULT, cmds)


def gen_acceleration(path):
    """Сценарий 2: Разгон 0 → 40 км/ч (20 с)"""
    # motion_type=2: приращение скорости 11.11 м/с с макс. ускорением 1.0
    cmds = ["2,0,0,0,11.11,0,0,20,1"]
    write_profile(path, INIT_DEFAULT, cmds)


def gen_straight_cruise(path):
    """Сценарий 3: Крейсер 40 км/ч (60 с)"""
    # init = {**INIT_CRUISE}
    # Стартуем сразу на скорости — задаём через ini vx_body
    # cmds = [
    #     "1,0,0,0,11.11,0,0,1,1",  # выходим на режим
    #     "5,0,0,0,11.11,0,0,60,1",  # держим
    # ]
    # Перепишем начальное состояние со скоростью
    # with open(path, "w", encoding="utf-8") as f:
    #     f.write(HEADER.replace("0,0,0,{yaw}", "11.11,0,0,{yaw}").format(**init))
    #     for c in cmds:
    #         f.write(c + "\n")

    cmds = ["1,0,0,0,0,0,0,60,1"]  # держим
    write_profile(path, INIT_CRUISE, cmds)


def gen_braking(path):
    """Сценарий 4: Торможение 40 → 0 км/ч (15 с)"""
    init = {**INIT_DEFAULT}
    cmds = ["2,0,0,0,-11.11,0,0,15,1"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER.replace("0,0,0,{yaw}", "11.11,0,0,{yaw}").format(**init))
        for c in cmds:
            f.write(c + "\n")


def gen_park_accel_cruise_brake(
    path,
    cruise_dur: float = 5.0,
    brake_dur: float = 15,
    target_kmh: float = 40.0,
):
    """Сценарий 10: ____"""
    v = target_kmh / 3.6  # км/ч → м/с  (40 км/ч ≈ 11.1111 м/с)
    a = 0.8
    t_acc = v / a

    cmds = [
        # фаза 1: стоянка — нулевая скорость, удержание позы
        f"1,0,0,0,0,0,0,5,1",
        # фаза 2: разгон 0 → v  (type 2 меняет vx линейно до целевого значения)
        f"1,0,0,0,{a},0,0,{t_acc:.3f},1",
        # фаза 3: крейсер — держать v прямо
        f"1,0,0,0,0,0,0,{cruise_dur},1",
        # фаза 4: торможение v → 0
        f"1,0,0,0,{-a},0,0,{t_acc:.3f},1",
    ]

    write_profile(path, INIT_DEFAULT, cmds)


def gen_turn(path):
    """Сценарий 5: Поворот на 90° при v=20 км/ч (15 с)"""
    # На скорости 5.56 м/с поворот на 90° за 15 с → угловая скорость 6 °/с
    cmds = [
        "1,0,0,0,0,0,0,2,1",  # выход на режим
        "1,6,0,0,0,0,0,15,1",  # приращение курса +90°
        "1,0,0,0,0,0,0,5,1",
    ]
    # with open(path, "w", encoding="utf-8") as f:
    #     f.write(HEADER.replace("0,0,0,{yaw}", "5.56,0,0,{yaw}").format(**init))
    #     for c in cmds:
    #         f.write(c + "\n")
    write_profile(path, INIT_TURN, cmds)


def gen_uphill(path):
    """Сценарий 6: Подъём в горку (60 с)"""
    init = {**INIT_DEFAULT}
    cmds = [
        "1,0,8,0,6.94,0,0,3,1",  # выход на режим с pitch=+8°
        "5,0,8,0,6.94,0,0,55,1",  # подъём 25 км/ч на склоне 8°
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER.replace("0,0,0,{yaw}", "6.94,0,0,{yaw}").format(**init))
        for c in cmds:
            f.write(c + "\n")


def gen_downhill_turn(path):
    """Сценарий 7: Спуск с поворотом (50 с)"""
    init = {**INIT_DEFAULT, "pitch": -8.0}
    cmds = [
        "1,0,-8,0,4.17,0,0,2,1",  # старт: спуск 15 км/ч, pitch=-8°
        "3,60,0,0,4.17,0,0,30,1",  # поворот на 60° за 30 с
        "5,60,-8,0,4.17,0,0,18,1",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER.replace("0,0,0,{yaw}", "4.17,0,0,{yaw}").format(**init))
        for c in cmds:
            f.write(c + "\n")


def gen_full_mission(path):
    """Сценарий 99: Полная миссия (всё вместе)"""
    cmds = [
        # 1. Стоянка 15 с
        "5,0,0,0,0,0,0,15,1",
        # 2. Разгон 0 → 30 км/ч за 15 с
        "2,0,0,0,8.33,0,0,15,1",
        # 3. Прямая 20 с
        "5,0,0,0,8.33,0,0,20,1",
        # 4. Подъём — задаём pitch +6° за 5 с
        "3,0,6,0,8.33,0,0,5,1",
        # 5. Движение на подъёме 30 с
        "5,0,6,0,8.33,0,0,30,1",
        # 6. Выход на горизонт
        "3,0,-6,0,8.33,0,0,5,1",
        # 7. Поворот на 90° за 18 с (v=5 м/с после притормаживания)
        "2,0,0,0,-3.33,0,0,5,1",  # сброс до 18 км/ч
        "3,90,0,0,5.0,0,0,18,1",
        # 8. Спуск с поворотом
        "3,0,-5,0,5.0,0,0,4,1",
        "3,45,0,0,5.0,0,0,20,1",
        # 9. Торможение до остановки
        "3,0,5,0,5.0,0,0,4,1",  # выход на горизонт
        "2,0,0,0,-5.0,0,0,12,1",  # торможение
        # 10. Финальная стоянка
        "5,0,0,0,0,0,0,10,1",
    ]
    write_profile(path, INIT_DEFAULT, cmds)


def generate_all(out_dir="motion_profiles"):
    os.makedirs(out_dir, exist_ok=True)
    gen_stationary(f"{out_dir}/01_stationary.csv")
    gen_acceleration(f"{out_dir}/02_acceleration.csv")
    gen_straight_cruise(f"{out_dir}/03_straight_cruise.csv")
    gen_braking(f"{out_dir}/04_braking.csv")
    gen_turn(f"{out_dir}/05_turn.csv")
    gen_uphill(f"{out_dir}/06_uphill.csv")
    gen_downhill_turn(f"{out_dir}/07_downhill_turn.csv")
    gen_park_accel_cruise_brake(f"{out_dir}/10_accel_cruise_brake.csv")
    gen_full_mission(f"{out_dir}/99_full_mission.csv")
    print(f"Сгенерировано 9 профилей в {out_dir}/")
