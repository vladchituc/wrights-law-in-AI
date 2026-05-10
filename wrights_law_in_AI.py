"""
Wright's Law in AI — final figure
Vlad Chituc, May 2026

Three-panel figure showing:
A. METR task horizon over time (Moore's-Law form for capability)
B. Cumulative frontier training compute over time (input)
C. Task horizon vs cumulative compute (Wright's-Law form, power-law fit)

Data sources:
- METR Time Horizon 1.1: https://metr.org/assets/benchmark_results_1_1.yaml
- Epoch AI frontier models: https://epoch.ai/data/ai-models

Method:
- Panels A, B: OLS on ln(y) vs date2num(release_date)
- Panel C: OLS on ln(horizon) vs ln(cumFLOP)  =  power-law fit
- Same n=17 SOTA METR models in all three panels
"""
import os
import yaml, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.dates import date2num
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# === Paths (resolve relative to this file so the script is reproducible) ===
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# === Fonts (Montserrat preferred; falls back to default sans-serif if not found) ===
for f in ['/tmp/Montserrat-Regular.ttf','/tmp/Montserrat-Bold.ttf','/tmp/Montserrat-Medium.ttf']:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
font_family = 'Montserrat' if any('Montserrat' in n.name for n in fm.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({
    'axes.spines.right':False,'axes.spines.top':False,
    'font.family':font_family,'font.weight':'regular','axes.titleweight':'regular',
    'figure.facecolor':'white','axes.facecolor':'white',
})

# === Time-axis tick array (METR's logarithmic_ticks, in minutes) ===
LOG_TICKS=np.array([1/60,2/60,4/60,8/60,15/60,30/60,1,2,4,8,15,30,60,2*60,4*60,8*60,16*60],dtype=float)
def fmt_time(seconds):
    seconds=round(seconds); hours=seconds/3600
    if hours>=1: return f"{int(hours)}h" if hours==int(hours) else f"{hours:.1f}h"
    if hours>=1/60: return f"{int(hours*60)}m"
    return f"{int(seconds)}s"
def fmt_doubling(v):
    if v < 1000: return f"{int(v)}×"
    if v < 1_000_000: return f"{int(v/1000)}K×"
    return f"{int(v/1_000_000)}M×"

# === Load METR (SOTA only, n=17) ===
with open(os.path.join(DATA_DIR, 'metr_benchmark.yaml')) as f: d=yaml.safe_load(f)
metr=pd.DataFrame([{
    'id':k,'date':pd.Timestamp(v['release_date']),
    'y':v['metrics']['p50_horizon_length']['estimate'],
    'sota':v['metrics'].get('is_sota',False),
} for k,v in d['results'].items()]).sort_values('date').reset_index(drop=True)
metr=metr[metr.sota].reset_index(drop=True)

# === Load Epoch frontier corpus + cumulative FLOP ===
ep=pd.read_csv(os.path.join(DATA_DIR, 'frontier_ai_models.csv'),low_memory=False)
ep['date']=pd.to_datetime(ep['Publication date'],errors='coerce')
ep['flop']=pd.to_numeric(ep['Training compute (FLOP)'],errors='coerce')
ep=ep.dropna(subset=['date','flop'])

# === RL post-training adjustments to Epoch's published values ===
# Epoch's 'Training compute (FLOP)' is mostly pre-training only; for reasoning-class
# models RL is up to ~50% of pre-training and should be added for "total compute".
# References: Epoch blog "How far can reasoning models scale?", DeepSeek-R1 paper analysis,
# Grok 4 entry (which already includes RL via geometric mean).
# Adjustments applied below as multipliers on Epoch's pre-training number.
RL_TOTAL_MULTIPLIER = {
    'Claude 3.7 Sonnet': 1.5,   # pre-only in Epoch → multiply for ~50% RL
    'GPT-5': 1.3,               # Epoch's value may include some RL; modest bump
}
for model, mult in RL_TOTAL_MULTIPLIER.items():
    mask = ep['Model']==model
    ep.loc[mask, 'flop'] = ep.loc[mask, 'flop'] * mult

# === Imputed FLOP for closed frontier-class models that Epoch flags as ">1e25 FLOP"
# but doesn't publish a precise estimate for. Values are TOTAL compute (pre + RL post-training).
# Sources: Epoch blog posts, OpenAI/Anthropic/Google statements, consistency with neighbors.
IMPUTED_FRONTIER = pd.DataFrame([
    # Pre-METR-set major closed releases (mostly pre-training, RL ~1-5%)
    ('Gemini 1.5 Pro',      '2024-02-15', 5.0e24),  # MoE, conservative estimate
    ('Claude 3 Opus',       '2024-03-04', 1.5e25),  # frontier Anthropic, GPT-4 era
    ('GPT-4 Turbo',         '2024-04-09', 2.1e25),  # ~same as GPT-4 base
    ('GPT-4o',              '2024-05-13', 2.0e25),  # incremental over GPT-4
    # Closed reasoning era 2024-2025 (RL ~10-50% of pre-training, included in totals)
    ('o1-preview',          '2024-09-12', 2.5e25),  # GPT-4o pre + small RL
    ('Claude 3.5 Sonnet (new)','2024-10-22', 3.0e25),  # similar pre to Jun 2024
    ('o1',                  '2024-12-05', 3.5e25),  # pre + ~10-20% RL
    ('o3',                  '2025-04-16', 8.0e25),  # OpenAI "10× beyond o1" → ~50% RL on top
    # Late-2025 / early-2026 frontier (significant RL, included in totals)
    ('Gemini 3 Pro',        '2025-11-18', 1.0e26),  # frontier Google TPU
    ('Claude Opus 4.5',     '2025-11-24', 8.0e25),  # larger Anthropic + RL
    ('GPT-5.2',             '2025-12-11', 1.1e26),  # incremental over GPT-5 + RL
    ('GPT-5.3 Codex',       '2026-02-05', 6.0e25),  # codex variant + RL
    ('Claude Opus 4.6',     '2026-02-05', 1.0e26),  # incremental over 4.5
    ('Gemini 3.1 Pro',      '2026-02-19', 1.4e26),  # incremental
    ('GPT-5.4',             '2026-03-05', 1.2e26),  # incremental
], columns=['Model','date','flop'])
IMPUTED_FRONTIER['date'] = pd.to_datetime(IMPUTED_FRONTIER['date'])

# Merge Epoch (FLOP-published with RL adjustments) with imputed (total compute estimates)
ep_all=pd.concat([ep[['Model','date','flop']], IMPUTED_FRONTIER],
                 ignore_index=True).sort_values('date').reset_index(drop=True)
metr['flop_cum']=metr['date'].apply(lambda d: ep_all[ep_all['date']<=d]['flop'].sum())

ANCHOR_CUM=1e22  # 1× = 1e22 cumulative FLOP

# === Fits ===
def fit_lin(x,y):
    X=np.asarray(x,float).reshape(-1,1); ly=np.log(np.clip(np.asarray(y,float),1e-3,np.inf))
    return LinearRegression().fit(X,ly)
def fit_pow(x,y):
    lx=np.log(np.asarray(x,float)); ly=np.log(np.asarray(y,float))
    a,lnA=np.polyfit(lx,ly,1); A=np.exp(lnA)
    pred=lnA+a*lx; r2=1-np.sum((ly-pred)**2)/np.sum((ly-ly.mean())**2)
    return A,a,r2

metr_t=np.array([date2num(d) for d in metr['date']])
regA=fit_lin(metr_t, metr['y'].values); slopeA=regA.coef_[0]; dblA=np.log(2)/slopeA
ly_a = np.log(metr['y'].values); ly_p = regA.predict(metr_t.reshape(-1,1))
r2A = 1 - np.sum((ly_a-ly_p)**2)/np.sum((ly_a-ly_a.mean())**2)
regB=fit_lin(metr_t, metr['flop_cum'].values); slopeB=regB.coef_[0]; dblB=np.log(2)/slopeB
ly_a = np.log(metr['flop_cum'].values); ly_p = regB.predict(metr_t.reshape(-1,1))
r2B = 1 - np.sum((ly_a-ly_p)**2)/np.sum((ly_a-ly_a.mean())**2)
A_w,a_w,r2_w = fit_pow(metr['flop_cum'].values, metr['y'].values)
A_rel = A_w * (ANCHOR_CUM ** a_w)

# === Style ===
BLUE='#2196f3'; LABEL='#444'; WMARK='#bbb'; TEXT_DARK='#222'
TITLE_FS=22; AXIS_FS=20; TICK_FS=17; LABEL_FS=17; ANNOT_FS=17
SUPTITLE_FS=28; WMARK_FS=16
DEX=6.0

HEADLINE_LABELS = {
    'gpt2':'GPT-2','davinci_002':'GPT 3','gpt_3_5_turbo_instruct':'GPT 3.5',
    'gpt_4':'GPT-4','gpt_4o_inspect':'GPT 4o','o1_inspect':'o1',
    'claude_opus_4_5_inspect':'Claude Opus 4.5',
}

def label_with_bg(ax, x, y, text, off):
    """geom_label_repel-style label with white background pill."""
    ax.annotate(text, xy=(x,y), xytext=off, textcoords='offset points',
                fontsize=LABEL_FS, color=TEXT_DARK, fontweight='medium',
                va='center', zorder=8,
                bbox=dict(facecolor='white', edgecolor='#dddddd',
                          boxstyle='round,pad=0.25', linewidth=0.8, alpha=0.95))

def metr_panel(ax, df, ycol, reg, title, watermark_right, legend_label,
               ylim, ykind, anchor=None, label_pos=None):
    xlim=(pd.Timestamp('2018-09-01'), pd.Timestamp('2026-09-01'))
    xs_num=np.linspace(date2num(xlim[0]), date2num(xlim[1]), 250)
    xs_dt=mdates.num2date(xs_num)
    pred=np.exp(reg.predict(xs_num.reshape(-1,1)))
    if anchor: pred=pred/anchor
    dmin,dmax=df['date'].min(),df['date'].max()
    inside=(xs_num>=date2num(dmin))&(xs_num<=date2num(dmax))
    ax.plot([d for d,m in zip(xs_dt,inside) if m], pred[inside], color=BLUE, lw=2.6, zorder=4)
    ax.plot([d for d,m in zip(xs_dt,inside) if not m], pred[~inside], color=BLUE, lw=2.6, ls='--', alpha=0.85, zorder=4)
    yvals = df[ycol].values / anchor if anchor else df[ycol].values
    ax.scatter(df['date'], yvals, s=140, color=BLUE, edgecolor='white', linewidth=1.2, zorder=6)
    for key, txt in HEADLINE_LABELS.items():
        sub=df[df['id']==key]
        if len(sub):
            r=sub.iloc[0]
            yv = r[ycol]/anchor if anchor else r[ycol]
            off = (label_pos or {}).get(key, (12,-3))
            label_with_bg(ax, r['date'], yv, txt, off)
    ax.set_yscale('log')
    if ykind=='time':
        t=LOG_TICKS[(LOG_TICKS>=ylim[0])&(LOG_TICKS<=ylim[1])]
        ax.set_yticks(t); ax.set_yticklabels([fmt_time(x*60) for x in t])
    elif ykind=='doublings':
        all_pows=[2**k for k in range(0,25)]
        t=[v for v in all_pows if v>=ylim[0] and v<=ylim[1]]
        labels=[fmt_doubling(v) if i%2==0 else '' for i,v in enumerate(t)]
        ax.set_yticks(t); ax.set_yticklabels(labels)
    ax.set_ylim(*ylim); ax.set_xlim(xlim[0],xlim[1])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(False)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999'); ax.spines['bottom'].set_color('#999')
    ax.tick_params(colors=TEXT_DARK, labelsize=TICK_FS)
    for tl in ax.get_xticklabels()+ax.get_yticklabels():
        tl.set_fontweight('medium')
    ax.set_xlabel('Model release date', fontsize=AXIS_FS, labelpad=12, color=TEXT_DARK, fontweight='medium')
    ax.text(0.0,1.06, title, transform=ax.transAxes, fontsize=TITLE_FS, fontweight='bold', color='#222', ha='left', va='bottom')
    ax.text(0.97,0.04, legend_label, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=ANNOT_FS, color=TEXT_DARK, fontweight='medium',
            bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.6', linewidth=0.8))
    ax.text(0.0,-0.18,'CC-BY', transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='left', va='top', fontweight='medium')
    ax.text(1.0,-0.18, watermark_right, transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='right', va='top', fontweight='medium')

# === Per-panel label offsets ===
labelsA = {'gpt2':(12,-3),'davinci_002':(14,-3),'gpt_3_5_turbo_instruct':(14,-3),
           'gpt_4':(-58,-15),'gpt_4o_inspect':(-78,-15),'o1_inspect':(14,3),
           'claude_opus_4_5_inspect':(-180,-3)}
labelsB = {'gpt2':(12,-3),'davinci_002':(14,-3),'gpt_3_5_turbo_instruct':(14,3),
           'gpt_4':(-58,-15),'gpt_4o_inspect':(-78,15),'o1_inspect':(14,3),
           'claude_opus_4_5_inspect':(-180,-3)}
labelsC = {'gpt2':(15,-3),'davinci_002':(15,-3),'gpt_3_5_turbo_instruct':(15,-3),
           'gpt_4':(-72,12),'gpt_4o_inspect':(-78,-15),'o1_inspect':(15,3),
           'claude_opus_4_5_inspect':(-180,-3)}

# === Build figure ===
fig, ax3 = plt.subplots(1, 3, figsize=(28, 9.5))
plt.subplots_adjust(top=0.78, bottom=0.13, left=0.04, right=0.985, wspace=0.22)

ylim_h=(1/200, 1/200*10**DEX)
metr_panel(ax3[0], metr, 'y', regA,
    'A. Task horizons double every 6 months',
    'metr.org', f'Doubling time: {dblA:.0f} days\nR² = {r2A:.2f}',
    ylim=ylim_h, ykind='time', label_pos=labelsA)

ylim_c=(1, 10**DEX)
metr_panel(ax3[1], metr, 'flop_cum', regB,
    f'B. Cumulative compute doubles every {dblB/30.4:.0f} months',
    'epoch.ai', f'Doubling time: {dblB:.0f} days\nR² = {r2B:.2f}',
    ylim=ylim_c, ykind='doublings',
    anchor=ANCHOR_CUM, label_pos=labelsB)

# Panel C: Wright's-law form (log-log power-law)
metr['rel_cum']=metr['flop_cum']/ANCHOR_CUM
ax=ax3[2]
xs=np.geomspace(1, 10**DEX, 250)
ax.plot(xs, A_rel*xs**a_w, color=BLUE, lw=2.6, zorder=5)
ax.scatter(metr['rel_cum'], metr['y'], s=140, color=BLUE, edgecolor='white', lw=1.2, zorder=6)
for key, txt in HEADLINE_LABELS.items():
    sub=metr[metr['id']==key]
    if len(sub):
        r=sub.iloc[0]
        off = labelsC.get(key,(15,-3))
        label_with_bg(ax, r['rel_cum'], r['y'], txt, off)
ax.set_xscale('log'); ax.set_yscale('log')
all_pows=[2**k for k in range(0,25)]
xt=[v for v in all_pows if v>=1 and v<=10**DEX]
xt_labels=[fmt_doubling(v) if i%2==0 else '' for i,v in enumerate(xt)]
ax.set_xticks(xt); ax.set_xticklabels(xt_labels)
yt=LOG_TICKS[(LOG_TICKS>=1/60)&(LOG_TICKS<=16*60)]
ax.set_yticks(yt); ax.set_yticklabels([fmt_time(t*60) for t in yt])
ax.set_xlim(1, 10**DEX); ax.set_ylim(1/60, 16*60)
ax.grid(False)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.spines['left'].set_color('#999'); ax.spines['bottom'].set_color('#999')
ax.tick_params(colors=TEXT_DARK, labelsize=TICK_FS)
for tl in ax.get_xticklabels()+ax.get_yticklabels():
    tl.set_fontweight('medium')
ax.set_xlabel('Relative cumulative compute', fontsize=AXIS_FS, labelpad=12, color=TEXT_DARK, fontweight='medium')
ax.text(0.0,1.06, f'C. Doubling task horizon needs {2**(1/a_w):.1f}× more compute',
        transform=ax.transAxes, fontsize=TITLE_FS, fontweight='bold', color='#222', ha='left', va='bottom')
ax.text(0.97,0.04, f'horizon $\\propto$ cumFLOP$^{{{a_w:.2f}}}$\nR² = {r2_w:.2f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOT_FS, color=TEXT_DARK, fontweight='medium',
        bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.6', linewidth=0.8))
ax.text(0.0,-0.18,'CC-BY', transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='left', va='top', fontweight='medium')
ax.text(1.0,-0.18,'metr.org · epoch.ai', transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='right', va='top', fontweight='medium')

# === Title (two-text with measured offset to avoid mathtext apostrophe issue) ===
fig.canvas.draw()
bold_t = fig.text(0.04, 0.965, "AI progress follows Wright's law:",
                  fontsize=SUPTITLE_FS, fontweight='bold', color='#222',
                  ha='left', va='top', family='Montserrat')
renderer = fig.canvas.get_renderer()
bbox = bold_t.get_window_extent(renderer=renderer)
fig_w_pix = fig.get_figwidth() * fig.dpi
x_after = (bbox.x1 / fig_w_pix) + 0.005
fig.text(x_after, 0.965,
         "exponential investment and diminishing returns create an exponential function in time",
         fontsize=SUPTITLE_FS, fontweight='regular', color='#333',
         ha='left', va='top', family='Montserrat')

out_path = os.path.join(FIG_DIR, 'wrights_law_in_AI.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'saved: {out_path}')

# === Print final fit numbers ===
r2A = 1 - np.sum((np.log(metr['y'].values) - regA.predict(metr_t.reshape(-1,1)))**2) / \
          np.sum((np.log(metr['y'].values) - np.log(metr['y'].values).mean())**2)
r2B = 1 - np.sum((np.log(metr['flop_cum'].values) - regB.predict(metr_t.reshape(-1,1)))**2) / \
          np.sum((np.log(metr['flop_cum'].values) - np.log(metr['flop_cum'].values).mean())**2)
print(f"\nFINAL FIT NUMBERS (n=17 METR SOTA models)")
print(f"Panel A: horizon doubling = {dblA:.0f} d ({dblA/30.4:.1f} mo), R²={r2A:.3f}")
print(f"Panel B: cumFLOP doubling = {dblB:.0f} d ({dblB/30.4:.1f} mo), R²={r2B:.3f}")
print(f"Panel C: horizon = {A_w:.2e} * cumFLOP^{a_w:.3f}, R²={r2_w:.3f}")
print(f"  → 2× horizon needs {2**(1/a_w):.2f}× cumulative compute")
print(f"  → 10× cumulative compute → {10**a_w:.2f}× horizon")
