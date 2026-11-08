"""Independent severity-effect thresholds; hypothetical requirements, not efficacy estimates."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import analysis as core
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'06_Analysis/results'; FIG=ROOT/'04_Figures'

def factors(p):
    d=sum((1-p['m'])**y*(1-p['m']/2)/(1+p['discount'])**(y+.5) for y in range(2))
    dental=p['initial']+sum(p['maint']*(1-p['m'])**t/(1+p['discount'])**t for t in [.5,1,2])
    cs=p['ch'] if p['composite'] else p['h']*p['ch']+(1-p['h'])*p['ed']
    return d,dental,cs

def independent(p,am,asev):
    d,dental,cs=factors(p)
    dc=dental-d*(p['rm']*p['cm']*am+p['rs']*cs*asev)
    dq=d*(p['rm']*p['dm']*am+p['rs']*p['ds']*asev)*p['duration']/365
    return float(dc),float(dq)

def threshold(p,am,wtp):
    d,dental,cs=factors(p)
    return (dental-d*p['rm']*(p['cm']+wtp*p['dm']*p['duration']/365)*am)/(d*p['rs']*(cs+wtp*p['ds']*p['duration']/365))

if __name__=='__main__':
    rows=[];grid=[];checks=[]
    reduction=1-core.run()['rr']
    for source in ['Tianjin','Chongqing']:
        for maint in [112,500,1032]:
            p={**core.BASE,**(core.chongqing() if source=='Chongqing' else {}),'maint':maint}
            for am in [0.,reduction]:
                row=dict(source=source,maintenance=maint,moderate_reduction=am)
                for w in [0,core.WTP,2*core.WTP]:
                    t=threshold(p,am,w);row['threshold_'+str(int(w))]=float(t)
                    dc,dq=independent(p,am,t)
                    assert np.isclose(w*dq-dc,0,atol=1e-8)
                rows.append(row)
            for am in np.linspace(0,1,101):
                for asev in np.linspace(0,1,101):
                    dc,dq=independent(p,am,asev)
                    grid.append(dict(source=source,maintenance=maint,moderate_reduction=am,severe_reduction=asev,dc=dc,dq=dq,inmb=core.WTP*dq-dc))
            for b in [0,.5,1]:
                a=independent(p,reduction,reduction*b);bresult=core.model({**p,'severe_effect':b})
                assert np.allclose(a,[bresult['dc'],bresult['dq']],atol=1e-10)
            dc,dq=independent(p,0,0);assert dq==0 and np.isclose(dc,factors(p)[1])
    checks.extend(['36 analytic threshold roots verified numerically','18 independent-kernel results match retained core','Six no-effect scenarios equal dental costs','Grid includes independent 0–100% moderate and severe reductions'])
    utility=[]
    for source in ['Tianjin','Chongqing']:
        for u,label in [(.90,'Primary: Dou 2025 Chinese-valued urban survey mean'),(.726,'Wu 2015 Japanese-valued mean'),(.768,'Legacy Wu 2015 Japanese-valued median'),(.74,'Sun 2026 Chinese-valued other triple-therapy mean'),(.62,'Sun 2026 Chinese-valued symptomatic SET mean')]:
            p={**core.BASE,**(core.chongqing() if source=='Chongqing' else {}),'u':u}
            r=core.model(p)
            utility.append(dict(source=source,utility=u,label=label,ucq=float(r['ucq']),trq=float(r['trq']),dc=float(r['dc']),dq=float(r['dq']),inmb=float(r['inmb'])))
            ref=core.model({**p,'u':.768})
            assert np.allclose([r[k] for k in ['dc','dq','inmb']],[ref[k] for k in ['dc','dq','inmb']])
    checks.append('Ten baseline-utility runs: incremental costs/QALYs/INMB unchanged')
    pd.DataFrame(rows).to_csv(OUT/'independent_effect_thresholds.csv',index=False)
    pd.DataFrame(grid).to_csv(OUT/'independent_effect_grid.csv',index=False)
    pd.DataFrame(utility).to_csv(OUT/'chinese_utility_scenarios.csv',index=False)
    colors=['#0072B2','#D55E00','#009E73'];x=np.linspace(0,1,101)
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,2,figsize=(9,4.5),sharey=True)
    for ax,source in zip(axes,['Tianjin','Chongqing']):
        for maint,color in zip([112,500,1032],colors):
            p={**core.BASE,**(core.chongqing() if source=='Chongqing' else {}),'maint':maint}
            for w,style in [(0,'--'),(core.WTP,'-')]:
                y=[100*threshold(p,v,w) for v in x]
                ax.plot(x*100,y,color=color,linestyle=style,lw=1.8)
        ax.set(title=source,xlabel='Assumed moderate-event reduction (%)',xlim=(0,100),ylim=(0,70))
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Required severe-event reduction (%)')
    handles=[Line2D([0],[0],color=c,lw=2,label=f'Maintenance CNY {m}/visit') for m,c in zip([112,500,1032],colors)]
    handles += [Line2D([0],[0],color='black',ls='--',label='Cost neutral'),Line2D([0],[0],color='black',label='CE at CNY 99,665/QALY')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8.5)
    fig.tight_layout(rect=(0,.16,1,1))
    for ext,dpi in [('png',180),('pdf',300),('tiff',300)]:
        fig.savefig(FIG/f'Figure3_Independent_Thresholds.{ext}',dpi=dpi,**({'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}))
    fig.savefig(FIG/'Figure3_Independent_Thresholds_1000dpi.tiff',dpi=1000,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    summary=dict(run_at_utc=datetime.now(timezone.utc).isoformat(),type='deterministic structural analysis; no empirical efficacy or structural probabilities',rows=rows,utility_rows=utility,grid_rows=len(grid),checks=checks,core_tests=core.tests(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),core_sha256=hashlib.sha256((ROOT/'06_Analysis/analysis.py').read_bytes()).hexdigest(),main_PSA='Rerun on 20 October 2026; see run_metadata.json and summary.json')
    (OUT/'threshold_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
