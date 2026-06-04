import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Импорт Times New Roman
_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/times.ttf",  # WSL → Windows
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",  # Linux msttcore
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

# Общие параметры для всех графиков
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 20,
        "axes.xmargin": 0,
        "legend.fontsize": 16,
        "legend.framealpha": 0.95,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# plt.rcParams.update(
#     {
#         "figure.facecolor": "white",
#         "axes.facecolor": "white",
#         "axes.grid": True,
#         "grid.alpha": 0.35,
#         "grid.linestyle": "--",
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "font.size": 20,
#         "axes.titlesize": 24,
#         "axes.titleweight": "bold",
#         "axes.labelsize": 20,
#         "axes.xmargin": 0,
#         "legend.fontsize": 14,
#         "legend.framealpha": 0.95,
#         "figure.dpi": 110,
#         "savefig.dpi": 200,
#         "savefig.bbox": "tight",
#     }
# )

# Единая палитра методов
METHOD_STYLE = {
    "Опорная": dict(color="k", lw=2.0, z=1),
    "EKF": dict(color="r", lw=1.5, z=3),
    "UKF": dict(color="g", lw=1.5, z=3),
    "FGO": dict(color="b", lw=1.5, z=3),
    "БИНС": dict(color="#9D4EDD", lw=1.5, z=2),
}
