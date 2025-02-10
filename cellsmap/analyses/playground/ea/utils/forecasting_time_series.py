import numpy as np


def recovery_rate(x_i, x_f, dt):
    return (x_f - x_i) / 2*dt

def recovery_rate_curve(x, dt):
    return x[1:-1], np.array([recovery_rate(x[i-1], x[i+1], dt) for i in range(1,len(x)-1)])