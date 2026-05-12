"""
The METR analog of the abstract confound illustration.

Assume the underlying Wright exponent is b = 0.60 (industrial-pack territory),
and that linear scaffolding growth gives each model a different prefactor a_i.
Fit those a_i to the actual METR data (with c = 0) and plot:

  - Each model's "true" Wright curve y = a_i · F^0.60 (a family of parallel lines
    on log-log axes, one per METR model)
  - The 17 actual METR data points (one on each curve)
  - The observed trajectory across models (black dashed line through the points)

The observed trajectory has a much steeper apparent slope than any individual
b = 0.60 curve, exactly because a_i grows across models. This is the empirical
version of the confound illustration applied to real data.
"""
import os
import yaml, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.cm import viridis
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

for f in ['/tmp/Montserrat-Regular.ttf', '/tmp/Montserrat-Bold.ttf', '/tmp/Montserrat-Medium.ttf']:
    if os.path.exists(f): fm.fontManager.addfont(f)
font_family = 'Montserrat' if any('Montserrat' in n.name for n in fm.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({
    'axes.spines.right': False, 'axes.spines.top': False,
    'font.family': font_family, 'font.weight': 'regular',
    'axes.titleweight': 'regular',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# ============================================================
# Load METR + cumulative compute (matches main analysis)
# ============================================================
with open(os.path.join(DATA_DIR, 'metr_benchmark.yaml')) as f: d = yaml.safe_load(f)
metr = pd.DataFrame([{
    'id': k, 'date': pd.Timestamp(v['release_date']),
    'y': v['metrics']['p80_horizon_length']['estimate'],
    'sota': v['metrics'].get('is_sota', False),
} for k, v in d['results'].items()]).sort_values('date').reset_index(drop=True)
metr = metr[metr.sota].sort_values('date').reset_index(drop=True)

ep = pd.read_csv(os.path.join(DATA_DIR, 'frontier_ai_models.csv'), low_memory=False)
ep['date'] = pd.to_datetime(ep['Publication date'], errors='coerce')
ep['flop'] = pd.to_numeric(ep['Training compute (FLOP)'], errors='coerce')
ep = ep.dropna(subset=['date', 'flop'])
RL_MULT = {'Claude 3.7 Sonnet': 1.5, 'GPT-5': 1.3}
for m, k in RL_MULT.items():
    ep.loc[ep['Model']==m, 'flop'] *= k
IMPUTED = pd.DataFrame([
    ('Gemini 1.5 Pro','2024-02-15',5.0e24),('Claude 3 Opus','2024-03-04',1.5e25),
    ('GPT-4 Turbo','2024-04-09',2.1e25),('GPT-4o','2024-05-13',2.0e25),
    ('o1-preview','2024-09-12',2.5e25),('Claude 3.5 Sonnet (new)','2024-10-22',3.0e25),
    ('o1','2024-12-05',3.5e25),('o3','2025-04-16',8.0e25),
    ('Gemini 3 Pro','2025-11-18',1.0e26),('Claude Opus 4.5','2025-11-24',8.0e25),
    ('GPT-5.2','2025-12-11',1.1e26),('GPT-5.3 Codex','2026-02-05',6.0e25),
    ('Claude Opus 4.6','2026-02-05',1.0e26),('Gemini 3.1 Pro','2026-02-19',1.4e26),
    ('GPT-5.4','2026-03-05',1.2e26),
], columns=['Model','date','flop'])
IMPUTED['date'] = pd.to_datetime(IMPUTED['date'])
ep_all = pd.concat([ep[['Model','date','flop']], IMPUTED], ignore_index=True)
metr['flop_cum'] = metr['date'].apply(lambda d: ep_all[ep_all['date']<=d]['flop'].sum())
t0 = metr['date'].min()
metr['t_years'] = (metr['date'] - t0).dt.days / 365.25

t = metr['t_years'].values
F = metr['flop_cum'].values
y = metr['y'].values
log_y = np.log(y)
y_var = np.sum((log_y - log_y.mean())**2)

# ============================================================
# Fit y = (α₀ + α₁·t) · F^b   at b = 0.60, c = 0
# (linear-in-time prefactor; no exponential allowed)
# ============================================================
B_FIXED = 0.60
def fit_c0_linear_a(b_fixed):
    def loss(lp):
        α0, α1 = np.exp(lp)
        a_t = α0 + α1 * t
        pred = a_t * F ** b_fixed
        return np.sum((np.log(np.maximum(pred, 1e-30)) - log_y) ** 2)
    a_init = np.log(y[0] / F[0]**b_fixed)
    best = None
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x0 = np.array([a_init, a_init]) + rng.normal(0, 1, size=2)
        res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 2000})
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x), 1 - best.fun / y_var

(α0, α1), r2 = fit_c0_linear_a(B_FIXED)
a_t_implied = α0 + α1 * t
ratio = a_t_implied[-1] / a_t_implied[0]
print(f"Fit at b = {B_FIXED}, c = 0, linear a:")
print(f"  α₀ = {α0:.2e}, α₁ = {α1:.2e}")
print(f"  a(t=0) = {a_t_implied[0]:.2e}")
print(f"  a(t=7) = {a_t_implied[-1]:.2e}")
print(f"  total growth: {ratio:.2f}×")
print(f"  R²(log-y) = {r2:.3f}")

# Apparent log-log slope across the data
apparent_slope, _ = np.polyfit(np.log10(F), np.log10(y), 1)
print(f"\nApparent log-log slope of observed trajectory: {apparent_slope:.2f}")
print(f"True per-model Wright exponent: {B_FIXED}")
print(f"Inflation factor: {apparent_slope/B_FIXED:.2f}×")

# ============================================================
# Plot — one panel, log-log, in the style of the confound figure
# ============================================================
TEXT = '#222'

HEADLINE_LABELS = {
    'gpt2': 'GPT-2',
    'davinci_002': 'GPT 3',
    'gpt_3_5_turbo_instruct': 'GPT 3.5',
    'gpt_4': 'GPT-4',
    'gpt_4o_inspect': 'GPT 4o',
    'o1_inspect': 'o1',
    'claude_opus_4_5_inspect': 'Claude Opus 4.5',
}

fig, ax = plt.subplots(figsize=(15, 9))
plt.subplots_adjust(top=0.78, bottom=0.10, left=0.07, right=0.97)

# Plot each METR model's own b=0.60 Wright curve, but only as a short segment
# centered on its data point. This makes the family of curves visible without
# crowding the figure with overlapping parallel lines.
F_extent = (np.log10(F.max()) - np.log10(F.min()))   # log-decades spanned
seg_dex = F_extent * 0.25                            # each segment spans ~1/4 of the x-range
for i in range(len(metr)):
    color = viridis(i / (len(metr) - 1))
    a_i = a_t_implied[i]
    log_F_center = np.log10(F[i])
    F_seg = np.logspace(log_F_center - seg_dex/2, log_F_center + seg_dex/2, 60)
    y_seg = a_i * F_seg ** B_FIXED
    ax.plot(F_seg, y_seg, color=color, lw=2.2, alpha=0.92, zorder=3)

# Apparent power-law fit line through the data (the "observed" log-log slope)
F_grid = np.geomspace(F.min() * 0.6, F.max() * 1.6, 200)
slope_obs, intercept_obs = np.polyfit(np.log10(F), np.log10(y), 1)
y_obs_line = 10**(intercept_obs + slope_obs * np.log10(F_grid))
ax.plot(F_grid, y_obs_line, color='black', lw=1.5, linestyle=':',
        alpha=0.55, zorder=4)

# Observed trajectory: black dashed connecting the data points
ax.plot(F, y, color='black', lw=2.4, linestyle='--', zorder=5)
ax.scatter(F, y, color='black', s=85, zorder=6, edgecolor='white', linewidth=1.2)

# Headline labels
for idx, row in metr.iterrows():
    if row['id'] in HEADLINE_LABELS:
        ax.annotate(HEADLINE_LABELS[row['id']],
                    xy=(row['flop_cum'], row['y']),
                    xytext=(12, -4), textcoords='offset points',
                    fontsize=12, color=TEXT, fontweight='medium',
                    bbox=dict(facecolor='white', edgecolor='#dddddd',
                              boxstyle='round,pad=0.25', linewidth=0.7, alpha=0.94),
                    zorder=8)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Cumulative compute (FLOP)', fontsize=13)
ax.set_ylabel('Horizon (minutes)', fontsize=13)
ax.tick_params(labelsize=11)

# Title above the plot
ax.text(0, 1.16,
        f"Moving target on real METR data: each model on its own Wright curve (b held at {B_FIXED:.2f})",
        transform=ax.transAxes, fontsize=15, fontweight='bold', color='#222',
        ha='left', va='top')
ax.text(0, 1.10,
        f"Each colored line is one METR model's hypothetical b=0.60 Wright curve, with its own prefactor a_i.\n"
        f"Black dashed = observed trajectory across models. Apparent log-log slope ≈ {apparent_slope:.2f}, "
        f"even though each underlying curve has b = {B_FIXED:.2f}.\n"
        f"The implied tooling factor a_t grows linearly from {a_t_implied[0]:.1e} to {a_t_implied[-1]:.1e} "
        f"({ratio:.1f}× total) — no exponential anywhere. Fit R²(log-y) = {r2:.3f}, vs. {0.909:.3f} for the unconstrained b=0.79 baseline.",
        transform=ax.transAxes, fontsize=10.5, color='#555',
        ha='left', va='top')

out_path = os.path.join(FIG_DIR, 'b_sensitivity_illustration.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nsaved: {out_path}")
