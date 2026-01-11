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

# ---------- Stimulus: DC + AC current (current clamp) ----------
def I_stim(t_ms, amp_dc=10.0, amp_ac=0.0, freq_ac=80.0, t_on=50.0, t_off=400.0):
    """DC step + AC sinusoidal background"""
    if t_on <= t_ms < t_off:
        ac = amp_ac * np.sin(2 * np.pi * freq_ac * t_ms / 1000.0)  # t_ms -> seconds
        return amp_dc + ac
    return 0.0

# ---------- RK4 solver ----------
def simulate_step(amp_dc, amp_ac=0.0, freq_ac=80.0, T=400.0, dt=0.025, V0=-65.0, t_on=50.0):
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
        I = I_stim(t, amp_dc=amp_dc, amp_ac=amp_ac, freq_ac=freq_ac, t_on=t_on, t_off=T)
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

# ---------- Build f-I curves with different AC backgrounds ----------
import os
import json
import shutil

amps_dc = np.arange(0.0, 50.0 + 1e-9, 0.05)  # uA/cm^2 (1000 points)
ac_amplitudes = np.array([10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0])  # 11 curves
frequencies = [20, 40, 60, 80, 100]  # Hz
SAVE_EVERY = 10  # Save checkpoint every N DC points

# ========== Google Drive Setup (Colab) ==========
USE_DRIVE = False
DRIVE_DIR = '/content/drive/MyDrive/HH_results'

try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(DRIVE_DIR, exist_ok=True)
    USE_DRIVE = True
    print(f"Google Drive mounted. Results will sync to: {DRIVE_DIR}")
except:
    print("Not on Colab or Drive not available. Using local storage only.")

def sync_to_drive(local_path):
    """Copy file to Google Drive for persistence"""
    if USE_DRIVE and os.path.exists(local_path):
        shutil.copy2(local_path, DRIVE_DIR)

# ========== Checkpoint System ==========
CHECKPOINT_FILE = 'checkpoint.json'
PARTIAL_DIR = 'partial_results'
os.makedirs(PARTIAL_DIR, exist_ok=True)

def load_checkpoint():
    # Try loading from Drive first
    if USE_DRIVE:
        drive_ckpt = f"{DRIVE_DIR}/checkpoint.json"
        if os.path.exists(drive_ckpt):
            shutil.copy2(drive_ckpt, CHECKPOINT_FILE)
            print("Restored checkpoint from Drive")
        # Restore partial results from Drive
        drive_partial = f"{DRIVE_DIR}/partial_results"
        if os.path.exists(drive_partial):
            for f in os.listdir(drive_partial):
                src = f"{drive_partial}/{f}"
                dst = f"{PARTIAL_DIR}/{f}"
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            print("Restored partial results from Drive")

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'progress': {}}

def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f)
    sync_to_drive(CHECKPOINT_FILE)

def get_progress_key(freq, ac_idx):
    return f"{freq}_{ac_idx}"

def load_partial(freq, ac_idx):
    """Load partial DC progress for a specific (freq, ac) pair"""
    path = f"{PARTIAL_DIR}/partial_{freq}Hz_ac{ac_idx}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d['rates'].tolist(), int(d['dc_idx'])
    return [], 0

def save_partial_dc(freq, ac_idx, rates, dc_idx):
    """Save partial DC progress"""
    path = f"{PARTIAL_DIR}/partial_{freq}Hz_ac{ac_idx}.npz"
    np.savez(path, rates=np.array(rates), dc_idx=dc_idx)
    sync_to_drive(path)
    # Also sync to Drive partial dir
    if USE_DRIVE:
        drive_partial = f"{DRIVE_DIR}/partial_results"
        os.makedirs(drive_partial, exist_ok=True)
        shutil.copy2(path, f"{drive_partial}/partial_{freq}Hz_ac{ac_idx}.npz")

def save_final(freq, results):
    """Save final results for a frequency"""
    data = {'amps_dc': amps_dc, 'ac_amplitudes': ac_amplitudes, 'freq_ac': freq}
    for i, amp_ac in enumerate(ac_amplitudes):
        data[f'rates_{i}'] = results[amp_ac]
    path = f'f_I_data_{freq}Hz.npz'
    np.savez(path, **data)
    sync_to_drive(path)

checkpoint = load_checkpoint()
print(f"Checkpoint: {len(checkpoint['completed'])} curves done, {len(checkpoint.get('progress', {}))} in progress")

for freq_ac in frequencies:
    print(f"\n=== Processing {freq_ac}Hz ===")
    results = {}

    for i, amp_ac in enumerate(ac_amplitudes):
        key = get_progress_key(freq_ac, i)

        # Skip if fully completed
        if key in checkpoint['completed']:
            # Load completed data
            rates, _ = load_partial(freq_ac, i)
            if len(rates) == len(amps_dc):
                print(f"  AC={amp_ac:.3g}: skipped (done)")
                results[amp_ac] = np.array(rates)
                continue

        # Load partial progress or start fresh
        rates, start_idx = load_partial(freq_ac, i)
        if start_idx > 0:
            print(f"  AC={amp_ac:.3g}: resuming from DC[{start_idx}]")

        # Continue from where we left off
        for j in tqdm(range(start_idx, len(amps_dc)), desc=f"AC={amp_ac:.3g}", initial=start_idx, total=len(amps_dc)):
            a = amps_dc[j]
            ts, V = simulate_step(a, amp_ac=amp_ac, freq_ac=freq_ac, T=1000.0, dt=0.01, V0=-65.0, t_on=50.0)
            r, _ = firing_rate_from_trace(ts, V, t_start=300.0, thresh=0.0)
            rates.append(r)

            # Save every SAVE_EVERY points
            if (j + 1) % SAVE_EVERY == 0:
                save_partial_dc(freq_ac, i, rates, j + 1)

        # Mark as completed
        results[amp_ac] = np.array(rates)
        save_partial_dc(freq_ac, i, rates, len(amps_dc))
        if key not in checkpoint['completed']:
            checkpoint['completed'].append(key)
        checkpoint['progress'].pop(key, None)
        save_checkpoint(checkpoint)
        print(f"  AC={amp_ac:.3g}: done")

    # Save final data for this frequency
    save_final(freq_ac, results)
    print(f"Saved: f_I_data_{freq_ac}Hz.npz")

    # Plot
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(ac_amplitudes)))
    for i, amp_ac in enumerate(ac_amplitudes):
        label = f"{amp_ac:.3g}" if amp_ac > 0 else "0"
        plt.plot(amps_dc, results[amp_ac], linewidth=1, color=cmap[i], label=label)

    plt.xlabel("DC current amplitude (µA/cm²)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"HH model f-I curve with {freq_ac}Hz AC background")
    plt.legend(loc='upper left', ncol=2, fontsize=8, title="AC (µA/cm²)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = f'f_I_curve_{freq_ac}Hz.png'
    plt.savefig(fig_path, dpi=150)
    sync_to_drive(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

print("\n=== All done! ===")
