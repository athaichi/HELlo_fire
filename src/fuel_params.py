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
    },
    2: { # Tractor fire break (set close to soybeans but inflamable)
        "w_0": 0.0, # no fuel (aka unburnable)
        "delta": 1.0, 
        "M_x": 0.3, # does this matter?
        "sigma": 2000, 
        "h": 0, # no heeat content
        "S_T": 0.0, "S_e": 0.0, # no spread
        "p_p": 32.
    }
}
