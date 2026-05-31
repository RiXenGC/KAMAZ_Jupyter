# Моделирование и комплексирование БИНС/СНС для беспилотного карьерного самосвала


P.S.: Работа выполняется в среде WSL/Ubuntu, управление зависимостями — [uv](https://docs.astral.sh/uv/).

## Содержание работы

1. **Моделирование БИНС.** Генерация инерциальных и спутниковых измерений в
   [Aceinna GNSS-INS-SIM](https://github.com/Aceinna/gnss-ins-sim), счисление
   ориентации фильтром Маджвика, Махони. Механизация в навигационной СК NED.
2. **Комплексирование ИНС/СНС.** Слабосвязанная (loosely-coupled) схема комплексирования
   через EKF, UKF и оптимизацию на факторном графе (FGO, на базе GTSAM).
3. **Сравнение методов**

---

## Структура проекта

```
KAMAZ_Jupyter/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
│
├── src
│   ├── config/                 
│   │   ├── constants.py        
│   │   └── imu_config.py       
│   │
│   ├── notebooks/
│   │   └── exploration.ipynb
│   │
│   ├── simulation/             # Группа 1 — моделирование измерений
│   │   ├── motion_profiles.py  #   генераторы профилей движения (CSV-команды)
│   │   ├── run_simulation.py   #   обёртка над GNSS-INS-SIM
│   │   ├── vibration.py        #   модель вибрации силовой установки
│   │   └── data_loader.py      #   чтение IMU + GPS + reference из выхода симулятора
│   │
│   ├── navigation/             # Группа 2 — счисление и комплексирование
│   │   ├── geodesy.py          #   радиусы кривизны, гравитация, скорость Земли (NED)
│   │   ├── attitude.py         #   кватернионы, матрицы поворота, углы Эйлера
│   │   ├── sins.py             #   БИНС-baseline (Маджвик + механизация)
│   │   ├── ekf.py              #   InsGnssEKF — расширенный фильтр Калмана
│   │   ├── ukf.py              #   InsGnssUKF — сигма-точечный фильтр
│   │   └── fgo.py              #   InsGnssFGO — оптимизация на факторном графе (GTSAM)
│   │
│   └── analysis/               # БЛОК 3 — анализ и визуализация
│       ├── frames.py           #   подготовка сценария: перевод в NED, шумы, вибрация
│       ├── metrics.py          #   метрики 
│       ├── visualization.py    #   вывод графиков
│       └── allanDev.py         #   построение девиации Аллана для модели ИИМ
│
├── scripts/                    # Пайплайн для запуска из терминала
│   ├── 01_generate_profiles.py #   создать все CSV-профили движения
│   ├── 02_run_simulation.py    #   прогнать симулятор → generated_data/
│   ├── 03_run_filters.py       #   EKF/UKF/FGO по сценариям → reports/
│   └── 04_compare_methods.py   #   сводная таблица и сравнительные графики
│
├── motion_profiles/            # сгенерированные CSV-профили (НЕ в git)
├── generated_data/             # выход симулятора (НЕ в git)
└── reports/                    # графики и таблицы (НЕ в git)
```

### Логика разделения на блоки

| Блок | Отвечает за | Зависит от |
|------|-------------|-----------|
| `simulation` | *генерацию* данных: профили, прогон симулятора, вибрация, загрузка | `config` |
| `navigation` | *оценку* состояния из данных: счисление, EKF/UKF/FGO | `config` |
| `analysis` | *сравнение и показ*: перевод в NED, метрики, графики | `simulation`, `navigation` |

---

## Установка

Требуется [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Развёртывание проекта:

```bash
git clone <repo-url> KAMAZ_Jupyter
cd my_prj_NIR
uv sync
```

Проверка установки на основе библиотеки GTSAM:

```bash
uv run python -c "import nir; import gtsam; print('OK, ImuFactor:', hasattr(gtsam,'ImuFactor'))"
```

---

## Зависимости

| Пакет | Назначение |
|-------|-----------|
| `numpy`, `scipy` | численное ядро, интегрирование, оптимизация |
| `pandas` | чтение CSV-выхода симулятора |
| `gnss-ins-sim` | генерация инерциальных и спутниковых измерений |
| `ahrs` | фильтр Маджвика (счисление ориентации) |
| `filterpy` | базовые классы EKF/UKF |
| `gtsam` | факторные графы, предынтегрирование IMU |
| `matplotlib` | визуализация |
| `jupyter`, `ipykernel` | ноутбуки (dev-зависимость) |

---

## Запуск

### Из терминала

```bash
uv run python scripts/01_generate_profiles.py   # создать профили движения
uv run python scripts/02_run_simulation.py      # прогнать симулятор
uv run python scripts/03_run_filters.py         # EKF/UKF/FGO по сценариям
uv run python scripts/04_compare_methods.py     # сводное сравнение
```

### Из ноутбука

...

---

## Профили движения

Генерируются в `motion_profiles/` как CSV-команды для GNSS-INS-SIM:

| Профиль | Описание |
|---------|----------|
| `01_stationary` | стоянка |
| `02_acceleration` | разгон 0 → 40 км/ч |
| `03_straight_cruise` | крейсерское движение |
| `04_braking` | торможение |
| `05_turn` | поворот на 90° |
| `06_uphill` | подъём на склоне |
| `07_downhill_turn` | спуск с поворотом |
| `10_accel_cruise_brake` | стоянка → разгон → крейсер → торможение |
| `99_full_mission` | полная миссия |

---

## Системы координат

Все вычисления ведутся в навигационной СК **NED** (North-East-Down),
связанная СК — **FRD** (Forward-Right-Down). Гравитация направлена по `+Z`
(вниз), угловая скорость вращения Земли `ω_ie^n = [Ω·cosφ, 0, −Ω·sinφ]`.
Положение в векторе состояния хранится в локальных метрах от точки старта.


## Воспроизводимость

В git коммитятся `pyproject.toml`, `uv.lock`, `.python-version` — этого
достаточно для точного воссоздания окружения. Тяжёлые артефакты
(`generated_data/`, `reports/`, `.venv/`) в git не попадают.