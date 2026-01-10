import numpy as np

# ---------- HH parameters (classic squid axon) ----------
C_m  = 1.0      # uF/cm^2
gNa  = 120.0    # mS/cm^2
gK   = 36.0
gL   = 0.3
ENa  = 50.0     # mV
EK   = -77.0
EL   = -54.387

def vtrap(x, y):
    # helps avoid 0/0 when x is small: x/(1-exp(-x/y))
    if abs(x / y) < 1e-6:
        return y * (1 - x / (2*y))
    return x / (1 - np.exp(-x / y))

# alpha/beta with V in mV
def alpha_n(V): return 0.01 * vtrap(V + 55.0, 10.0)
def beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)

def alpha_m(V): return 0.1 * vtrap(V + 40.0, 10.0)
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def dm_dt(V, m): return alpha_m(V)*(1-m) - beta_m(V)*m
def dh_dt(V, h): return alpha_h(V)*(1-h) - beta_h(V)*h
def dn_dt(V, n): return alpha_n(V)*(1-n) - beta_n(V)*n

def currents(V, m, h, n):
    INa = gNa * (m**3) * h * (V - ENa)
    IK  = gK  * (n**4)       * (V - EK)
    IL  = gL               * (V - EL)
    return INa, IK, IL

# ---------- stimulus templates ----------
def I_step(t, amp=10.0, t_on=10.0, t_off=40.0):
    return amp if (t_on <= t < t_off) else 0.0

def I_pulse_train(t, amp=10.0, t_start=10.0, n_pulses=5, period=10.0, width=2.0):
    if t < t_start: return 0.0
    k = int((t - t_start) // period)
    if k < 0 or k >= n_pulses: return 0.0
    phase = (t - t_start) - k*period
    return amp if (0.0 <= phase < width) else 0.0

def I_ramp(t, amp0=0.0, amp1=15.0, t0=10.0, t1=40.0):
    if t < t0: return 0.0
    if t >= t1: return amp1
    return amp0 + (amp1-amp0)*(t - t0)/(t1 - t0)

def I_sine(t, amp=5.0, freq_hz=20.0, t_on=0.0):
    # freq in Hz; t in ms -> convert: sin(2π f t_sec)
    if t < t_on: return 0.0
    return amp * np.sin(2*np.pi*freq_hz*(t/1000.0))

# ---------- RK4 integrator ----------
def simulate(I_fn, T=50.0, dt=0.01, V0=-65.0):
    ts = np.arange(0.0, T+dt, dt)
    V  = np.zeros_like(ts)
    m  = np.zeros_like(ts)
    h  = np.zeros_like(ts)
    n  = np.zeros_like(ts)
    Iapp = np.zeros_like(ts)

    # initialize gating at steady-state for V0
    def x_inf(alpha, beta, V): 
        a = alpha(V); b = beta(V)
        return a/(a+b)

    V[0] = V0
    m[0] = x_inf(alpha_m, beta_m, V0)
    h[0] = x_inf(alpha_h, beta_h, V0)
    n[0] = x_inf(alpha_n, beta_n, V0)

    INa = np.zeros_like(ts); IK = np.zeros_like(ts); IL = np.zeros_like(ts)

    def f(state, t):
        Vv, mm, hh, nn = state
        I = I_fn(t)
        ina, ik, il = currents(Vv, mm, hh, nn)
        dV = (I - (ina + ik + il)) / C_m
        dm = dm_dt(Vv, mm)
        dh = dh_dt(Vv, hh)
        dn = dn_dt(Vv, nn)
        return np.array([dV, dm, dh, dn]), (ina, ik, il, I)

    state = np.array([V[0], m[0], h[0], n[0]])
    for i in range(len(ts)-1):
        t = ts[i]

        k1, aux1 = f(state, t)
        k2, aux2 = f(state + 0.5*dt*k1, t + 0.5*dt)
        k3, aux3 = f(state + 0.5*dt*k2, t + 0.5*dt)
        k4, aux4 = f(state + dt*k3, t + dt)

        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V[i+1], m[i+1], h[i+1], n[i+1] = state

        # log currents using current state (or aux1; here recompute for consistency)
        ina, ik, il = currents(V[i+1], m[i+1], h[i+1], n[i+1])
        INa[i+1], IK[i+1], IL[i+1] = ina, ik, il
        Iapp[i+1] = I_fn(ts[i+1])

    return ts, V, m, h, n, Iapp, INa, IK, IL

# ---------- examples ----------
# 1) step current
ts, V, m, h, n, Iapp, INa, IK, IL = simulate(
    lambda t: I_step(t, amp=10.0, t_on=10.0, t_off=40.0),
    T=60.0, dt=0.01
)

# 2) pulse train (uncomment to run instead)
# ts, V, m, h, n, Iapp, INa, IK, IL = simulate(
#     lambda t: I_pulse_train(t, amp=8.0, t_start=10.0, n_pulses=6, period=10.0, width=1.0),
#     T=80.0, dt=0.01
# )
