import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Classic HH parameters (squid axon) ----------
C_m  = 1.0      # uF/cm^2
gNa  = 120.0    # mS/cm^2
gK   = 36.0
gL   = 0.3
ENa  = 50.0     # mV
EK   = -77.0
EL   = -54.387

def vtrap(x, y):
    # x/(1-exp(-x/y)) with a safe limit near 0
    z = x / y
    if abs(z) < 1e-6:
        return y * (1 + z/2)  # first-order expansion
    return x / (1 - np.exp(-z))

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

def x_inf(alpha, beta, V):
    a = alpha(V); b = beta(V)
    return a/(a+b)

# ---------- Stimulus: step current (current clamp) ----------
def I_step(t_ms, amp=10.0, t_on=50.0, t_off=400.0):
    return amp if (t_on <= t_ms < t_off) else 0.0

# ---------- RK4 solver ----------
def simulate_step(amp, T=400.0, dt=0.025, V0=-65.0, t_on=50.0):
    ts = np.arange(0.0, T+dt, dt)
    V  = np.zeros_like(ts)
    m  = np.zeros_like(ts)
    h  = np.zeros_like(ts)
    n  = np.zeros_like(ts)

    V[0] = V0
    m[0] = x_inf(alpha_m, beta_m, V0)
    h[0] = x_inf(alpha_h, beta_h, V0)
    n[0] = x_inf(alpha_n, beta_n, V0)

    def f(state, t):
        Vv, mm, hh, nn = state
        I = I_step(t, amp=amp, t_on=t_on, t_off=T)
        ina, ik, il = currents(Vv, mm, hh, nn)
        dV = (I - (ina + ik + il)) / C_m
        dm = dm_dt(Vv, mm)
        dh = dh_dt(Vv, hh)
        dn = dn_dt(Vv, nn)
        return np.array([dV, dm, dh, dn])

    state = np.array([V[0], m[0], h[0], n[0]])
    for i in range(len(ts)-1):
        t = ts[i]
        k1 = f(state, t)
        k2 = f(state + 0.5*dt*k1, t + 0.5*dt)
        k3 = f(state + 0.5*dt*k2, t + 0.5*dt)
        k4 = f(state + dt*k3, t + dt)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        V[i+1], m[i+1], h[i+1], n[i+1] = state

    return ts, V

def firing_rate_from_trace(ts, V, t_start=200.0, thresh=0.0):
    """
    Calculate firing rate using ISI (inter-spike interval) method.
    Returns mean frequency from steady-state ISIs for continuous values.
    """
    mask = ts >= t_start
    t = ts[mask]
    v = V[mask]

    # Find spike times (upward threshold crossings)
    crossing_idx = np.where((v[:-1] < thresh) & (v[1:] >= thresh))[0]
    n_spikes = len(crossing_idx)

    if n_spikes < 2:
        # Not enough spikes for ISI calculation
        # Fall back to counting method
        duration_s = (t[-1] - t[0]) / 1000.0 if len(t) > 1 else 0.0
        rate_hz = (n_spikes / duration_s) if duration_s > 0 else 0.0
        return rate_hz, n_spikes

    # Get spike times
    spike_times = t[crossing_idx]

    # Calculate ISIs (inter-spike intervals) in ms
    isis = np.diff(spike_times)

    # Use mean ISI for frequency (more stable than last ISI)
    mean_isi_ms = np.mean(isis)
    rate_hz = 1000.0 / mean_isi_ms  # Convert to Hz

    return rate_hz, n_spikes

# ---------- Build f-I curve ----------
amps = np.arange(0.0, 50.0 + 1e-9, 0.1)  # uA/cm^2 (500 points)
rates = []
spike_counts = []

for a in tqdm(amps, desc="Scanning I"):
    ts, V = simulate_step(a, T=1000.0, dt=0.01, V0=-65.0, t_on=50.0)
    r, nsp = firing_rate_from_trace(ts, V, t_start=300.0, thresh=0.0)
    rates.append(r); spike_counts.append(nsp)

rates = np.array(rates)
spike_counts = np.array(spike_counts)

# Estimate rheobase as first amp with at least 1 spike in measurement window
idx = np.where(spike_counts > 0)[0]
rheobase = amps[idx[0]] if len(idx) else None

# ---------- Plot ----------
plt.figure(figsize=(7,4.5))
plt.plot(amps, rates, linewidth=1.2)
plt.xlabel("Step current amplitude I (µA/cm²)")
plt.ylabel("Firing rate (Hz)")
plt.title("HH model f-I curve (step current clamp)")
plt.grid(True, alpha=0.3)
if rheobase is not None:
    plt.axvline(rheobase, linestyle='--')
    plt.text(rheobase+0.2, max(rates)*0.1, f"rheobase≈{rheobase:.1f}", rotation=90, va='bottom')
plt.tight_layout()
plt.savefig('f_I_curve.png', dpi=150)
print(f"Saved: f_I_curve.png")
print(f"Rheobase ≈ {rheobase:.1f} µA/cm²" if rheobase else "No spikes detected")
print(f"Max firing rate: {rates.max():.1f} Hz")
