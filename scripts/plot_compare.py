"""Top/bottom training-curve comparison: g1.cu (this repo) vs nanoG1, SAME GPU (RTX 3090).

Both curves come from REAL local training logs:
  - nanoG1: parsed from the PufferLib dashboard log (full_train.log) -- velocity-tracking
    score `perf` vs wall-clock. ~150M steps in ~5 min on the 3090 (SPS ~492K).
  - g1.cu : parsed from models/ppo_ckpt/reward_curve_fused.csv eval checkpoints -- forward
    distance toward the 5 m finish line vs wall-clock (the fused on-GPU PPO rollout).

The tasks DIFFER (nanoG1 = omnidirectional velocity-command tracking; g1.cu = walk and cross a
5 m finish line upright), so the metrics are not directly comparable -- hence stacked subplots
sharing only the wall-clock x-axis. Honest takeaway: both train a G1 humanoid to locomote from
scratch in single-digit minutes on ONE consumer GPU.

  uv run python scripts/plot_compare.py
"""
import os, re, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# nanoG1's curve is committed (bench/nanoG1_train_curve.csv) so this figure reproduces from a
# clean clone. To regenerate it from nanoG1's raw PufferLib log, set NANO_LOG to full_train.log.
NANO_CSV = os.path.join(HERE, "bench", "nanoG1_train_curve.csv")
NANO_LOG = os.environ.get("NANO_LOG", "")
OUR_CSV = os.path.join(HERE, "models", "ppo_ckpt", "reward_curve_fused.csv")
OUT = os.path.join(HERE, "demo", "train_compare.png")

# --- our run timing: fused PPO, H=32, N=8192; instantaneous sps lets us recover wall-clock ---
H, N = 32, 8192


def parse_ours():
    t, fwd, fall = [], [], []
    with open(OUR_CSV) as f:
        for row in csv.DictReader(f):
            if row["fwd_mean"] == "nan":
                continue  # eval runs every 25 iters; skip non-eval rows
            it = int(row["iter"]) ; sps = float(row["sps"])
            elapsed = (it + 1) * H * N / sps            # wall-clock seconds at this eval
            t.append(elapsed / 60.0)
            fwd.append(float(row["fwd_mean"]))
            fall.append(float(row["fall_rate"]))
    return t, fwd, fall


def parse_nano():
    """The PufferLib dashboard rewrites the same panel each refresh; split per refresh and pull
    (Uptime, Steps, perf). Uptime format is 'Nd Nh Nm Ns' (early sub-second ticks like '999ms'
    are skipped -- training has barely started there)."""
    txt = open(NANO_LOG).read()
    def to_sec(s):
        d = dict((u, int(v)) for v, u in re.findall(r"(\d+)([dhms])", s))
        return d.get("d",0)*86400 + d.get("h",0)*3600 + d.get("m",0)*60 + d.get("s",0)
    ups, steps, perf = [], [], []
    for b in txt.split("PufferLib"):                 # one block per dashboard refresh
        mu = re.search(r"Uptime\s+(\d+d\s+\d+h\s+\d+m\s+\d+s)", b)
        ms = re.search(r"Steps\s+([0-9.]+)M", b)
        mp = re.search(r"\bperf\s+([0-9.]+)", b)
        if mu and ms and mp:
            ups.append(to_sec(mu.group(1))/60.0); steps.append(float(ms.group(1))); perf.append(float(mp.group(1)))
    return ups, steps, perf


def load_nano():
    """Prefer the committed parsed CSV; regenerate from nanoG1's raw log if NANO_LOG is set."""
    if NANO_LOG and os.path.exists(NANO_LOG):
        nt, nsteps, nperf = parse_nano()
        os.makedirs(os.path.dirname(NANO_CSV), exist_ok=True)
        with open(NANO_CSV, "w") as f:
            f.write("# nanoG1 PufferLib training (RTX 3090), parsed from full_train.log\n")
            f.write("minutes,steps_M,perf\n")
            for a, b, c in zip(nt, nsteps, nperf):
                f.write(f"{a:.4f},{b:.2f},{c:.4f}\n")
        print(f"regenerated {NANO_CSV} ({len(nt)} pts)")
        return nt, nsteps, nperf
    nt, nsteps, nperf = [], [], []
    with open(NANO_CSV) as f:
        for row in f:
            if not row[:1].isdigit():
                continue
            a, b, c = row.split(",")
            nt.append(float(a)); nsteps.append(float(b)); nperf.append(float(c))
    return nt, nsteps, nperf


ot, ofwd, ofall = parse_ours()
nt, nsteps, nperf = load_nano()

# nanoG1 "walk" milestone: RESULTS.md gate at ~75M samples -> first time Steps>=75M on this 3090 log
nano_walk_t = next((nt[i] for i, s in enumerate(nsteps) if s >= 75.0), None)
# our "finish" milestone: first eval reaching the 5 m line
our_cross_t = next((ot[i] for i, d in enumerate(ofwd) if d >= 4.99), None)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
fig.suptitle("Training a Unitree G1 to locomote, from scratch, on ONE RTX 3090",
             fontsize=14, fontweight="bold")

# --- top: nanoG1 ---
ax1.plot(nt, nperf, color="#888888", lw=2, label="nanoG1: velocity-tracking score (perf)")
if nano_walk_t:
    ax1.axvline(nano_walk_t, color="#888888", ls="--", lw=1.3)
    ax1.annotate(f"walk gate (~75M steps)\n~{nano_walk_t:.1f} min",
                 xy=(nano_walk_t, 0.5), xytext=(nano_walk_t+0.3, 0.45),
                 fontsize=9, color="#555555")
ax1.set_ylabel("perf (vel-track, 0-1)")
ax1.set_title("nanoG1  -  omnidirectional velocity-command tracking", fontsize=11, loc="left")
ax1.set_ylim(0, 1.0); ax1.grid(alpha=0.3); ax1.legend(loc="lower right", fontsize=9)

# --- bottom: ours ---
ax2.plot(ot, ofwd, color="#1f77b4", lw=2, marker="o", ms=3,
         label="g1.cu: forward distance reached (m)")
ax2.axhline(5.0, color="#2ca02c", ls=":", lw=1.5, label="5 m finish line")
if our_cross_t:
    ax2.axvline(our_cross_t, color="#1f77b4", ls="--", lw=1.3)
    ax2.annotate(f"crosses finish line\n~{our_cross_t:.1f} min",
                 xy=(our_cross_t, 5.0), xytext=(our_cross_t+0.3, 3.2),
                 fontsize=9, color="#1f77b4")
ax2.set_ylabel("forward distance (m)")
ax2.set_xlabel("wall-clock training time (minutes, RTX 3090)")
ax2.set_title("g1.cu (this repo)  -  walk and cross a 5 m finish line upright", fontsize=11, loc="left")
ax2.set_ylim(0, 5.6); ax2.grid(alpha=0.3); ax2.legend(loc="lower right", fontsize=9)

fig.text(0.5, 0.012,
         "Same GPU, from-scratch RL, different tasks (metrics not directly comparable). "
         "g1.cu's sim runs ~1.57x nanoG1's physics throughput on an RTX PRO 6000.",
         ha="center", fontsize=8.5, color="#666666")
fig.tight_layout(rect=(0, 0.035, 1, 0.97))
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
print(f"  nanoG1: {len(nt)} pts, perf {nperf[0]:.2f}->{nperf[-1]:.2f}, last t={nt[-1]:.1f} min, walk@~{nano_walk_t:.1f} min")
print(f"  g1.cu : {len(ot)} pts, fwd {ofwd[0]:.2f}->{ofwd[-1]:.2f} m, cross@~{our_cross_t:.1f} min")
