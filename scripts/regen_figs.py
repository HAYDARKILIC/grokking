import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"serif","font.size":7,
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.4,
    "axes.spines.top":False,"axes.spines.right":False,
    "legend.frameon":False,"axes.linewidth":0.7,
    "xtick.major.width":0.7,"ytick.major.width":0.7,"savefig.dpi":600})

def boot(v,n=10000,seed=0):
    v=np.asarray([x for x in v if np.isfinite(x)],float)
    r=np.random.default_rng(seed)
    return tuple(np.quantile(r.choice(v,size=(n,len(v)),replace=True).mean(1),[.025,.975]))

df=pd.read_pickle("runs.pkl"); c=df[df.tag=="C"]
rows=[]
for b in sorted(c.beta.unique()):
    for s in sorted(c.seed.unique()):
        cp=c[(c.beta==b)&(c.seed==s)&(c.decay_variant=="coupled_l2")]
        dc=c[(c.beta==b)&(c.seed==s)&(c.decay_variant=="decoupled_wd")]
        if len(cp) and len(dc) and cp.iloc[0].grokked and dc.iloc[0].grokked:
            rows.append(dict(beta=b,ratio=dc.iloc[0].k_grok/cp.iloc[0].k_grok))
P=pd.DataFrame(rows)

# ================================================================= Fig 3
fig,(a0,a1)=plt.subplots(2,1,figsize=(3.10,4.55),
                         gridspec_kw=dict(height_ratios=[1.0,0.92],hspace=0.42))
# (a) measured against predicted, log-log
lim=[0.72,34]
a0.plot(lim,lim,ls="--",c="0.45",lw=0.9,label=r"identity $(1-\beta)^{-1}$",zorder=1)
for b,g in P.groupby("beta"):
    pred=1/(1-b); m=g.ratio.mean(); lo,hi=boot(g.ratio)
    a0.scatter(np.full(len(g),pred),g.ratio,s=9,c="tab:blue",alpha=0.55,zorder=2)
    a0.errorbar([pred],[m],yerr=[[m-lo],[hi-m]],fmt="o",ms=3.6,c="k",
                capsize=2.2,lw=0.9,zorder=4)
    a0.annotate(rf"$\beta$={b:g}",(pred,m),textcoords="offset points",
                xytext=(6,-7),fontsize=6.2)
a0.set_xscale("log"); a0.set_yscale("log"); a0.set_xlim(*lim); a0.set_ylim(*lim)
a0.set_xlabel(r"predicted ratio $(1-\beta)^{-1}$")
a0.set_ylabel(r"measured $k_{\mathrm{grok}}^{\mathrm{dec}}/k_{\mathrm{grok}}^{\mathrm{coup}}$")
a0.set_title("(a) measured against predicted",fontsize=7.4)
a0.legend(loc="upper left",fontsize=6.4,handlelength=1.6)

# (b) ratio to the prediction, which is where the identity breaks
betas=sorted(P.beta.unique()); x=np.arange(len(betas))
a1.axhspan(0.85,1.15,color="0.85",zorder=0)
a1.axhline(1.0,ls="--",c="0.45",lw=0.9,zorder=1)
for i,b in enumerate(betas):
    g=P[P.beta==b]; pred=1/(1-b)
    r=g.ratio/pred; m=r.mean(); lo,hi=boot(r)
    inside = lo<=1.0<=hi
    a1.scatter(np.full(len(r),i)+np.random.default_rng(1).uniform(-.10,.10,len(r)),
               r,s=9,c="tab:blue",alpha=0.55,zorder=2)
    a1.errorbar([i],[m],yerr=[[m-lo],[hi-m]],fmt="o" if inside else "s",
                ms=3.8,c="k" if inside else "tab:red",capsize=2.4,lw=1.0,zorder=4)
    a1.annotate(f"{100*(m-1):+.0f}%",(i,hi),textcoords="offset points",
                xytext=(0,4),ha="center",fontsize=6.2,
                color="0.3" if inside else "tab:red")
a1.set_xticks(x); a1.set_xticklabels([f"{b:g}" for b in betas])
a1.set_xlim(-0.5,len(betas)-0.5); a1.set_ylim(0.70,1.70)
a1.set_xlabel(r"momentum $\beta$")
a1.set_ylabel("measured / predicted")
a1.set_title("(b) departure from the identity",fontsize=7.4)
a1.text(0.03,0.035,r"grey band: $\pm15\%$ resolution of the $\beta{=}0$ null control",
        transform=a1.transAxes,fontsize=6.0,color="0.35")
fig.savefig("fig3_phase_c_p4.pdf",bbox_inches="tight"); plt.close(fig)

# ================================================================= Fig 4
sh=pd.read_pickle("ga_shifts.pkl")
g=sh.groupby("c")[["comp","acc"]].agg(["mean","std"])
C=g.index.to_numpy(float)
cm,cs=g[("comp","mean")].to_numpy(),g[("comp","std")].to_numpy()
am,asd=g[("acc","mean")].to_numpy(),g[("acc","std")].to_numpy()
R=5.5698e-4
fig,ax=plt.subplots(figsize=(3.10,2.95))
xs=np.linspace(C.min(),C.max(),300)
ax.plot(xs,np.log(xs)/R,ls="--",c="0.45",lw=0.9,zorder=1,
        label=r"prediction, slope $1/\hat r_{\mathrm{g}}$")
ax.errorbar(C,am,yerr=asd,fmt="x",ms=4,c="tab:red",capsize=1.8,lw=0.8,
            zorder=2,label=r"test accuracy, $k_{\mathrm{grok}}$")
ax.errorbar(C,cm,yerr=cs,fmt="o",ms=3.2,c="tab:blue",capsize=1.8,lw=0.8,
            zorder=3,label=r"component, $k_{\mathrm{comp}}$")
ax.set_xlabel(r"rescaling $c$ of the grokking component")
ax.set_ylabel("shift in crossing step")
ax.set_ylim(-108,108)
ax.legend(loc="upper left",fontsize=6.2,handlelength=1.6,labelspacing=0.3)

# inset: the five innermost rescalings, which overlap at the main scale
sel=np.abs(C-1)<=0.0051
ins=ax.inset_axes([0.575,0.065,0.405,0.355])
ins.plot(xs,np.log(xs)/R,ls="--",c="0.45",lw=0.8,zorder=1)
ins.errorbar(C[sel],am[sel],yerr=asd[sel],fmt="x",ms=3.2,c="tab:red",
             capsize=1.3,lw=0.7,zorder=2)
ins.errorbar(C[sel],cm[sel],yerr=cs[sel],fmt="o",ms=2.6,c="tab:blue",
             capsize=1.3,lw=0.7,zorder=3)
ins.set_xlim(0.9942,1.0058); ins.set_ylim(-13,13)
ins.set_xticks([0.995,1.0,1.005]); ins.set_xticklabels(["0.995","1","1.005"])
ins.set_yticks([-10,0,10])
ins.tick_params(labelsize=5.6,length=1.8,width=0.5)
for s in ins.spines.values(): s.set_linewidth(0.5); s.set_visible(True)
ins.grid(alpha=0.2,lw=0.3)
ax.indicate_inset_zoom(ins,edgecolor="0.55",lw=0.55,alpha=0.75)
fig.tight_layout(pad=0.25)
fig.savefig("fig4_phase_d_p2.pdf",bbox_inches="tight"); plt.close(fig)
print("fig3, fig4 rebuilt")
