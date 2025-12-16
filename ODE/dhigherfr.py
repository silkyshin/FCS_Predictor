# Model info
model_name = "1-2-4-8 dimer-higher order oligomer model"
kinetic_constants = ['k1','k_-1','k2','k_-2','k3','k_-3']

def ode_model(t, M, k):
    M1, M2, M4, M8 = M
    k1, k_1, k2, k_2, k3, k_3 = k

    dM1 = -2*k1*M1**2 + 2*k_1*M2
    dM2 = k1*M1**2 - k_1*M2 - 2*k2*M2**2 + 2*k_2*M4
    dM4 = k2*M2**2 - k_2*M4 - 2*k3*M4**2 + 2*k_3*M8
    dM8 = 2*k3*M4**2 - k_3*M8

    return [dM1,dM2,dM4,dM8]
