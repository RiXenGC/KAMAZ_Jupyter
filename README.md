# Моделирование и комплексирование БИНС/СНС для БКТС

## Содержание работы

1. **Моделирование БИНС.**

   Генерация ИИМ и ГНСС в [Aceinna GNSS-INS-SIM](https://github.com/Aceinna/gnss-ins-sim). Ориентация определяется фильтром Маджвиком, Махони. Механизация в NED.

2. **Комплексирование ИНС/СНС.** 

   Слабосвязанная схема комплексирования через EKF, UKF и оптимизацию на факторном графе (FGO, на базе [GTSAM](https://gtsam.org/)).

3. **Сравнение методов**

---

## Структура 

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
│   │   ├── motion_profiles.py  #   генерация профилей движения .csv
│   │   ├── run_simulation.py   #   обёртка над GNSS-INS-SIM
│   │   ├── vibration.py        #   модель вибраций
│   │   └── data_loader.py      #   чтение ИИМ + GPS + опорное
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
│   ├── 01_generate_profiles.py #   создать .csv
│   ├── 02_run_simulation.py    #   прогнать симулятор → generated_data/
│   ├── 03_run_filters.py       #   EKF/UKF/FGO по сценариям → reports/
│   └── 04_compare_methods.py   #   сводная таблица и сравнительные графики
│
├── motion_profiles/            # сгенерированные .csv
├── generated_data/             # выход симулятора 
└── reports/                    
```

### Логика 

| Группа | Что делает | 
|------|-------------|
| `simulation` | *Генерация* данных: профили, прогон симулятора, вибрация, загрузка | 
| `navigation` | *Оценка* состояния: счисление, EKF/UKF/FGO | 
| `analysis` | *Анализ результатов*: перевод в NED, метрики, графики |

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

## Профили движения

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
