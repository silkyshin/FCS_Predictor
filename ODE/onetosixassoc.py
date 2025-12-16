# Model info
model_name = "1-2-3-4-5-6 linear stepwise assuming later kn is identical"
kinetic_constants = ['k1','K1','k_assoc','K23','K34','K45','K56']

def ode_model(t, M, k):
    M1, M2, M3, M4, M5, M6 = M
    k1, K1, k_assoc, K23, K34, K45, K56 = k

    # Define reverse rates using equilibrium constants Kn
    k_1 = k1 / K1
    k_2 = k_assoc / K23
    k_3 = k_assoc / K34
    k_4 = k_assoc / K45
    k_5 = k_assoc / K56

    dM1 = -2*k1*M1**2 + 2*k_1*M2 - k_assoc*M1*M2 + k_2*M3 - k_assoc*M1*M3 + k_3*M4 - k_assoc*M1*M4 + k_4*M5 - k_assoc*M1*M5 + k_5*M6
    dM2 = k1*M1**2 - k_1*M2 - k_assoc*M1*M2 + k_2*M3
    dM3 = k_assoc*M1*M2 - k_2*M3 - k_assoc*M1*M3 + k_3*M4
    dM4 = k_assoc*M1*M3 - k_3*M4 - k_assoc*M1*M4 + k_4*M5
    dM5 = k_assoc*M1*M4 - k_4*M5 - k_assoc*M1*M5 + k_5*M6
    dM6 = k_assoc*M1*M5 - k_5*M6

    return [dM1,dM2,dM3,dM4,dM5,dM6]
