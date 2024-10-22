import numpy as np

def process_data(t_LT, uds_LT, samples=50001, T=1e-6):
    dt = T / samples
    t = np.linspace(0 * T, 1 * T - dt, samples)
    uds = np.interp(t, t_LT, uds_LT)
    
    return t, uds

def save_data(t, uds):
    np.savetxt('uds_Zeit.dat', [t, uds])  # Time domain signal