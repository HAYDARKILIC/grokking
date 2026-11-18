import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

W = 6.876   # == \textwidth (495.08 pt) : placed at width=\textwidth, scale 1.0
plt.rcParams.update({
    "font.family":"serif","font.size":8,
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.4,
    "axes.spines.top":False,"axes.spines.right":False,
    "legend.frameon":False,"axes.linewidth":0.7,
    "xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.labelsize":8,"ytick.labelsize":8,
    "axes.labelsize":8,"axes.titlesize":8.5,"legend.fontsize":8,
    "savefig.dpi":600})

rows=[json.loads(l) for l in open('/tmp/res/grokking_wd_clock/runs/runs.jsonl')]
A=[r for r in rows if r['tag']=='A'][0]; h=A['history']
st=np.array(h['steps'],float); tr=np.array(h['train_acc'],float)
te=np.array(h['test_acc'],float); nm=np.array(h['theta_norm'],float)

fig,axes=plt.subplots(1,2,figsize=(W,2.60))
ax=axes[0]
ax.plot(st,tr,lw=1.3,label="train"); ax.plot(st,te,lw=1.3,label="test")
ax.axhline(0.90,ls="--",c="0.45",lw=0.8)
for k,lab in ((A['k_mem'],r"$k_{\mathrm{mem}}$"),(A['k_grok'],r"$k_{\mathrm{grok}}$")):
    ax.axvline(k,ls=":",c="0.25",lw=0.8)
    ax.annotate(lab,(k,0.42),xytext=(4,0),textcoords="offset points",
                rotation=90,fontsize=8,va="center")
ax.set_xscale("log"); ax.set_xlabel(r"step $k$"); ax.set_ylabel("accuracy")
ax.set_title("(a) delayed generalization"); ax.legend(loc="center left")
ax=axes[1]
ax.plot(st,nm,lw=1.3,c="k",label=r"$\|\theta_k\|_2$")
w=(st>=1805)&(st<=9025)                      # the fitting window of eq. (19)
ax.axvspan(1805,9025,color="0.88",zorder=0,label="rate-fit window")
A=np.vstack([st[w],np.ones(w.sum())]).T
m,b=np.linalg.lstsq(A,np.log(nm[w]),rcond=None)[0]
yh=A@[m,b]; r2=1-((np.log(nm[w])-yh)**2).sum()/((np.log(nm[w])-np.log(nm[w]).mean())**2).sum()
ax.plot(st[w],np.exp(yh),ls="--",c="tab:red",lw=1.1,
        label=r"fit $\hat r=%.2f\times10^{-5}$, $R^{2}=%.3f$"%(-m*1e5,r2))
ax.set_yscale("log")
ax.set_xlabel(r"step $k$"); ax.set_ylabel(r"$\|\theta_k\|_2$")
ax.set_title("(b) contraction of the parameter norm")
ax.legend(loc="lower left",fontsize=7)
fig.suptitle(r"Reproduction of the reference benchmark "
             r"($p=17$, width $128$, $\beta=0.9$, $\eta=0.2$, $\lambda=0.0003$)",
             fontsize=9,y=1.01)
fig.tight_layout(pad=0.3)
fig.savefig("fig1_phase_a_reproduction.pdf",bbox_inches="tight"); plt.close(fig)

df=pd.read_pickle("runs.pkl")
B=df[(df.tag=="B")&(df.grokked.astype(bool))]
fig,axes=plt.subplots(1,2,figsize=(W,2.95),sharey=True)
for ax,p in zip(axes,[17,23]):
    sub=B[B.p==p]
    cells=sorted({(int(w),float(s)) for w,s in zip(sub.width,sub.init_scale)})
    data=[sub[(sub.width==w)&(sub.init_scale==s)].delay_ratio.to_numpy() for w,s in cells]
    pos=np.arange(1,len(cells)+1)
    ax.boxplot(data,positions=pos,widths=0.55,showfliers=False,
               medianprops=dict(color="k",lw=1.0),boxprops=dict(lw=0.8),
               whiskerprops=dict(lw=0.8),capprops=dict(lw=0.8))
    for x,d in zip(pos,data):
        ax.scatter(np.full(len(d),x)+np.random.default_rng(0).uniform(-.13,.13,len(d)),
                   d,s=11,c="tab:blue",alpha=0.6,zorder=3,edgecolors="none")
    ax.axhline(26,ls="--",c="tab:red",lw=1.0,
               label="value reported by the source" if p==17 else None)
    ax.set_yscale("log"); ax.set_xticks(pos)
    ax.set_xticklabels([f"{w}\n{s:g}" for w,s in cells])
    ax.set_title(rf"$p={p}$")
    ax.set_xlabel("width $H$ (upper) and initialization scale $s$ (lower)")
axes[0].set_ylabel(r"delay ratio $k_{\mathrm{grok}}/k_{\mathrm{mem}}$")
axes[0].legend(loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig("fig2_phase_b_variability.pdf",bbox_inches="tight"); plt.close(fig)
print("done")
