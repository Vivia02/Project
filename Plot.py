from matplotlib import pyplot as plt
import numpy as np

def plot_time_domain(t, uds, T, dt):
    plt.plot((t - t[0]) * 1e6, uds)
    plt.xlabel('t in µs')
    plt.ylabel('u_{ds} in V')
    plt.grid(visible=True)
    plt.xlim(0, (T - dt) * 1e6)
    plt.show()

