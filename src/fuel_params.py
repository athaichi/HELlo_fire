# fuel_params.py
import numpy as np

FUEL_MODELS = {
    1: { # Soybeans (treat like grass for now)
        "w_0": 1.0,     # plenty of fuel
        "delta": 1.0,   # shallow bed
        "M_x": 0.3,     # extinction moisture
        "sigma": 2000,  # fine fuel
        "h": 8000,      # heat content
        "S_T": 0.055,
        "S_e": 0.01,
        "p_p": 32,
    }
}
