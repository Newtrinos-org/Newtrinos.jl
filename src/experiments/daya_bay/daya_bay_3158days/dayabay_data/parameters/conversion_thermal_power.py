from scipy.constants import value

# reactor power conversion: Convert W[GW]/[MeV] N[fissions] to N[fissions]/T[s]
conversion_reactor_power = 1000 * value("joule-electron volt relationship")

configuration = {
    "format": "value",
    "state": "fixed",
    "parameters": {
        "conversion_reactor_power": conversion_reactor_power,
    },
    "labels": {
        "conversion_reactor_power": {
            "text": "Convert thermal power [GW/MeV]→[s⁻¹]",
            "latex": "Convert thermal power [GW/MeV]→[s$^{-1}$]",
        },
    },
}
