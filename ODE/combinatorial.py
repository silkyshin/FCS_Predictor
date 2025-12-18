# Model info
model_name = "1-2-3-4-5-6 combinatorial oligomerization"
kinetic_constants = ['k1','k_1','k2','k_2','k2','k_2','k3','k_3','k4','k_4','k5','k_5','k6','k_6','k7','k_7','k8','k_8','k9','k_9']

def ode_model(t, M, k):
    M1, M2, M3, M4, M5, M6 = M
    k1, k_1, k2, k_2, k2, k_2, k3, k_3, k4, k_4, k5, k_5, k6, k_6, k7, k_7, k8, k_8, k9, k_9 = k

    dM1 = -2*k1*M1**2 + 2*k_1*M2 - k2*M1*M2 + k_2*M3 - k3*M1*M3 + k_3*M4 - k4*M1*M4 + k_4*M5 - k5*M1*M5 + k_5*M6
    dM2 = k1*M1**2 - k_1*M2 - k2*M1*M2 + k_2*M3 - 2*k6*M2**2 + 2*k_6*M4 - k7*M2*M3 + k_7*M5 - k9*M2*M4 + k_9*M6
    dM3 = k2*M1*M2 - k_2*M3 - k3*M1*M3 + k_3*M4 - k7*M2*M3 + k_7*M5 - 2*k8*M3**2 + 2*k_8*M6
    dM4 = k3*M1*M3 - k_3*M4 - k4*M1*M4 + k_4*M5 + k6*M2**2 - k_6*M4 - k9*M2*M4 + k_9*M6
    dM5 = k4*M1*M4 - k_4*M5 - k5*M1*M5 + k_5*M6 + k7*M2*M3
    dM6 = k5*M1*M5 - k_5*M6 + k8*M3**2 - k_8*M6 + k9*M2*M4 - k_9*M6

    return [dM1,dM2,dM3,dM4,dM5,dM6]
