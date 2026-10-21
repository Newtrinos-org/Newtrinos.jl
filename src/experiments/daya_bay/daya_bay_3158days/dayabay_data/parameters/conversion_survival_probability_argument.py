from numpy import pi
from scipy.constants import value

eVkm = 1.0e-3 * value("electron volt-inverse meter relationship")
survival_probability_argument_factor = eVkm * 2 * pi

configuration = {
    "format": "value",
    "state": "fixed",
    "parameters": {
        "survival_probability_argument_factor": survival_probability_argument_factor,
    },
    "labels": {
        "survival_probability_argument_factor": {
            "text": "Convert Δm²·L/E from [eV²·km/MeV] to natural units",
            "latex": r"Convert $\Delta m^{2}$L/E from [eV$^{2}\cdot$km/MeV] to natural units",
        }
    },
}
