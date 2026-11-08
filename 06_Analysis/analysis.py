"""Exploratory COPD-periodontal decision model, revision 2026-10-20.

Published aggregate inputs are not patient data. Binary-to-rate mapping, dental
resources and cost transportability are assumptions, never fitted efficacy.
Run: python analysis.py --n 50000 --seed 20261015
"""
from pathlib import Path
import argparse, json, sys, platform, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if (HERE / '_deps').exists():
    sys.path.insert(0, str(HERE / '_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CPI = {2017:1.016,2018:1.021,2019:1.029,2020:1.025,2021:1.009,
       2022:1.020,2023:1.002,2024:1.002,2025:1.0}
WTP = 99665.0  # 2025 national GDP/person, NBS 2026 statistical communique.

def inflation(year, target=2024):
    return float(np.prod([v for k,v in CPI.items() if year < k <= target]))

# Dou 2025, Table 3: Chinese-valued EQ-5D-5L mean 0.90 (SD 0.13),
# matched urban survey n=361. Not a calibrated utility for the older target cohort.
# Wu 2015: 0.768 is the median; the mean is 0.726, Japanese tariff.
# Stable utility cancels exactly in all incremental outcomes under equal survival.
BASE = dict(m=1-np.exp(-.0517), rm=.17, rs=.31, u=.90, dm=.055, ds=.09,
            duration=14., discount=.045, or_=.29, p0=12/18,
            cm=341*inflation(2018), ch=17796*inflation(2018),
            initial=1032., maint=112., h=1., ed=341*inflation(2018),
            severe_effect=0., share=1., first_only=False, composite=False,
            maint_times=(.5,1.,2.))

def model(p):
    """Vectorized paired-strategy calculations; recurrent events while alive."""
    rr=p['or_']/(1-p['p0']+p['p0']*p['or_'])
    cs=p['ch'] if p['composite'] else p['h']*p['ch']+(1-p['h'])*p['ed']
    uc_m=uc_s=tr_m=tr_s=ucq=dq=em=es=am=asev=0.
    for y in range(2):
        exposure=(1-p['m'])**y*(1-p['m']/2)
        de=exposure/(1+p['discount'])**(y+.5)
        red=(1-rr)*p['share']*(0 if p['first_only'] and y==1 else 1)
        uc_m+=de*p['rm']*p['cm']; uc_s+=de*p['rs']*cs
        tr_m+=de*p['rm']*(1-red)*p['cm']
        tr_s+=de*p['rs']*(1-red*p['severe_effect'])*cs
        ucq+=de*(p['u']-(p['rm']*p['dm']+p['rs']*p['ds'])*p['duration']/365)
        dq+=de*red*(p['rm']*p['dm']+p['rs']*p['ds']*p['severe_effect'])*p['duration']/365
        em+=exposure*p['rm']; es+=exposure*p['rs']
        am+=exposure*p['rm']*red; asev+=exposure*p['rs']*red*p['severe_effect']
    maintenance=sum(p['maint']*(1-p['m'])**t/(1+p['discount'])**t for t in p['maint_times'])
    uc=uc_m+uc_s; tr=p['initial']+maintenance+tr_m+tr_s; dc=tr-uc
    return dict(uc=uc,tr=tr,dc=dc,dq=dq,ucq=ucq,trq=ucq+dq,
                uc_m=uc_m,uc_s=uc_s,tr_m=tr_m,tr_s=tr_s,initial=p['initial'],
                maintenance=maintenance,em=em,es=es,am=am,as_=asev,
                hospital_avoided=np.nan if p['composite'] else p['h']*asev,
                death=1-(1-p['m'])**2,inmb=WTP*dq-dc,rr=rr)

def run(**kw):
    return model({**BASE,**kw})

def wilson(x=12,n=18,z=1.959963984540054):
    q=x/n;den=1+z*z/n
    c=(q+z*z/(2*n))/den
    half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return c-half,c+half

def cost_moments(rho=.5):
    """Conditional ratio-estimator proxy, not a reported or validated cost SE."""
    rows=[]
    for name,a,sa,v,sv,unit in [('outpatient',2661,3415,7.91,8.31,341),
                               ('hospital',12857,31367,.72,1.22,17796)]:
        ratio=a/v
        se=np.sqrt(sa**2+ratio**2*sv**2-2*ratio*rho*sa*sv)/(np.sqrt(6759)*v)
        cv=se/ratio; totalcv=np.sqrt((1+cv**2)*1.04-1)
        rows.append(dict(cost=name,rho=rho,n_patients=6759,annual_cost_mean=a,
                         annual_cost_sd=sa,annual_visits_mean=v,annual_visits_sd=sv,
                         reported_unit_mean=unit,ratio_mean=ratio,ratio_se=se,
                         sampling_cv=cv,calibrated_se_2024=unit*inflation(2018)*cv,
                         total_cv=totalcv,gamma_shape=1/totalcv**2,
                         gamma_scale=unit*inflation(2018)*totalcv**2))
    return rows

def chongqing(year=2022):
    # Price year is NOT supplied in the article. 2022 is a declared scenario anchor.
    # Published severe cost already pools hospitalization/ER/death-defined episodes.
    return dict(cm=1022.5*inflation(year),ch=10215.7*inflation(year),composite=True)

def psa(severe=0,maint=112,n=50000,seed=20261015,cv=.2,rho=.5,
        p0=12/18,cost_source='tianjin',discount=.045):
    rng=np.random.default_rng(seed)
    def gamma(mean,coef):
        return np.full(n,mean) if coef==0 else rng.gamma(1/coef**2,mean*coef**2,n)
    costcv=[np.sqrt((1+r['sampling_cv']**2)*(1+cv**2)-1) for r in cost_moments(rho)]
    cp=chongqing() if cost_source=='chongqing' else {}
    if cp:costcv=[cv,cv]  # No event-independent SD/sqrt(n) construction.
    # SD 0.05 is an assumed transfer range, not the study SD or its mean SE.
    u=BASE['u']
    a=u*(u*(1-u)/.05**2-1)
    b=(1-u)*(u*(1-u)/.05**2-1)
    p={**BASE,**cp,'m':1-np.exp(-rng.uniform(.040,.065,n)),
       'rm':gamma(.17,(.18-.16)/3.92/.17),'rs':gamma(.31,(.33-.29)/3.92/.31),
       'u':rng.beta(a,b,n),'dm':rng.uniform(.0275,.0825,n),
       'ds':rng.uniform(.045,.135,n),'duration':rng.triangular(7,14,21,n),
       'discount':discount,'or_':rng.lognormal(np.log(.29),(np.log(.84)-np.log(.10))/3.92,n),
       'p0':p0,'cm':gamma(cp.get('cm',BASE['cm']),costcv[0]),
       'ch':gamma(cp.get('ch',BASE['ch']),costcv[1]),
       'initial':rng.triangular(774,1032,1290,n),
       'maint':rng.triangular(.75*maint,maint,1.25*maint,n),'severe_effect':severe}
    r=model(p);dc=r['dc'];dq=r['dq'];nmb=WTP*dq-dc
    s={k:[float(np.mean(x)),float(np.quantile(x,.025)),float(np.quantile(x,.975))]
       for k,x in [('dc',dc),('dq',dq),('inmb',nmb)]}
    prob=float(np.mean(nmb>0))
    s.update(p_cost_saving=float(np.mean(dc<0)),p1=prob,
             p2=float(np.mean(2*WTP*dq-dc>0)),
             evpi=float(np.mean(np.maximum(0,nmb))-max(0,np.mean(nmb))),
             n=n,seed=seed,mcse_probability=float(np.sqrt(prob*(1-prob)/n)),
             cost_source=cost_source,maint=maint,p0=p0,transfer_cv=cv)
    return s,dc,dq

def subgroup(k_n=3,k_f=3,prevalence=401/1235,severe=0,targeted=False,
             contrast='equal_severe',or_value=.29,maint=112):
    # Total-event category counts supply a hypothetical moderate-rate contrast.
    # The severe contrast is separated because hospitalization risks are similar.
    gm=((42+60*k_f)/299)/((56+24*k_n)/309)
    gs={'equal_severe':1.,'hospital_risk_proxy':(43/299)/(44/309),'shared_contrast':gm}[contrast]
    rn={};rf={}
    for key,g in [('rm',gm),('rs',gs)]:
        rn[key]=BASE[key]/(1-prevalence+prevalence*g);rf[key]=g*rn[key]
        assert np.isclose(prevalence*rf[key]+(1-prevalence)*rn[key],BASE[key])
    r_f=run(**rf,severe_effect=severe,or_=or_value,maint=maint)
    r_n=run(**rn,or_=1,initial=0 if targeted else BASE['initial'],maint=0 if targeted else maint)
    w={key:prevalence*r_f[key]+(1-prevalence)*r_n[key] for key in r_f}
    return dict(**w,p_frequent=prevalence,k_nonfrequent=k_n,k_frequent=k_f,
                moderate_contrast=gm,severe_contrast=gs,contrast=contrast,
                rm_frequent=rf['rm'],rs_frequent=rf['rs'],rm_nonfrequent=rn['rm'],rs_nonfrequent=rn['rs'],
                targeted=targeted,severe_effect=severe,or_value=or_value)

def scenario_specs():
    s=[('Primary moderate only',{}),('No COPD effect',{'or_':1}),
       ('Equal effects on both severities',{'severe_effect':1}),
       ('Severe reduction attenuated 50%',{'severe_effect':.5}),
       ('First-year-only effect; moderate only',{'first_only':True}),
       ('First-year-only effect; both severities',{'severe_effect':1,'first_only':True}),
       ('Benefit in 25% of treated cohort; both severities',{'severe_effect':1,'share':.25}),
       ('Benefit in 50% of treated cohort; both severities',{'severe_effect':1,'share':.5}),
       ('Upper OR moderate only',{'or_':.84}),
       ('Upper OR both severities',{'or_':.84,'severe_effect':1}),
       ('Scaling same assumed moderate effect',{'initial':112}),
       ('Scaling no COPD effect',{'initial':112,'or_':1}),
       ('Wu Japanese-valued mean utility 0.726',{'u':.726}),
       ('Legacy Wu median utility 0.768',{'u':.768}),
       ('Zero COPD QALY benefit cost-consequence',{'dm':0.,'ds':0.})]
    for h in [.25,.5,.75,(2735*.27+2237*.29)/(2735+2237)/.31]:
        s.append((f'Both severities hospital share {h:.3f}',dict(severe_effect=1,h=h)))
    for cost in [0,200,500,1000,1032]:
        s.append((f'Both severities maintenance {cost}',dict(severe_effect=1,maint=cost)))
    for d in [0,.05]:
        for sev in [0,1]:s.append((f'Discount {d}; severe effect {sev}',dict(discount=d,severe_effect=sev)))
    for sourceyear in [2016,2019]:
        s.append((f'Tianjin price-year assumption {sourceyear}; both severities',dict(severe_effect=1,cm=341*inflation(sourceyear),ch=17796*inflation(sourceyear))))
    for year in [2018,2020,2022,2023,2024]:
        for sev in [0,1]:
            s.append((f'Chongqing price-year anchor {year}; severe effect {sev}',dict(**chongqing(year),severe_effect=sev)))
    for v in wilson():
        for sev in [0,1]:s.append((f'Control risk {v:.6f}; severe effect {sev}',dict(p0=v,severe_effect=sev)))
    for sev in [0,1]:s.append((f'Legacy 5 percent and mortality approximation; severe {sev}',dict(m=.0517,discount=.05,severe_effect=sev)))
    return s

def tests():
    p=run();b=run(severe_effect=1);null=run(or_=1)
    assert np.isclose(p['death'],1-np.exp(-.0517*2))
    assert p['as_']==0 and p['dc']>0 and b['dc']<0
    assert null['dq']==0 and null['am']==0 and null['as_']==0
    assert np.isclose(null['dc'],null['initial']+null['maintenance'])
    for _,kw in scenario_specs():
        r=run(**kw)
        assert np.isclose(r['tr'],r['initial']+r['maintenance']+r['tr_m']+r['tr_s'])
        assert np.isclose(r['trq']-r['ucq'],r['dq'])
        assert np.isclose(r['tr']-r['uc'],r['dc'])
    assert np.isclose(run(m=.0517,discount=.05)['dc'],1279.5789005403278)
    # Exact continuous survival integral is an independent check of HCC approximation.
    h=.0517;d=np.log1p(BASE['discount'])
    exact=(1-np.exp(-2*(h+d)))/(h+d)
    approx=sum(np.exp(-h*y)*(1-BASE['m']/2)/(1+BASE['discount'])**(y+.5) for y in range(2))
    assert abs(approx-exact)/exact<.001
    s1,c1,q1=psa(n=1000);s2,c2,q2=psa(n=1000)
    assert np.array_equal(c1,c2) and np.array_equal(q1,q2)
    sv,cv,qv=psa(n=1000,severe=1)
    assert np.all((qv-q1)*q1>=-1e-14)
    assert np.all((cv-c1)*q1<=1e-10)
    # Vector/scalar consistency using identical explicit vectors.
    arr={k:np.array([v,v]) if isinstance(v,(int,float)) and not isinstance(v,bool) else v for k,v in BASE.items()}
    rv=model(arr)
    for k in ['dc','dq','uc','tr','ucq','trq']:assert np.allclose(rv[k],p[k])
    return dict(status='passed',checks=['legacy numerical reproduction','cost reconciliation','QALY reconciliation','null effect','no severe primary benefit','mortality rate conversion','HCC integral check','fixed-seed reproducibility','vector-scalar agreement'],hcc_relative_difference=float((approx-exact)/exact))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=50000)
    parser.add_argument('--seed',type=int,default=20261015);parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    figs=HERE.parent/'04_Figures';figs.mkdir(exist_ok=True)
    check=tests()
    det=[dict(scenario=name,**run(**kw)) for name,kw in scenario_specs()]
    pd.DataFrame(det).to_csv(out/'scenario_results.csv',index=False)
    pd.DataFrame([dict(scenario=f'h={h}; EDx{ed}; maintenance={m}',**run(severe_effect=1,h=h,ed=BASE['ed']*ed,maint=m))
                  for h in [.25,.5,.75,1] for ed in [1,3,5] for m in [112,500,1032]]).to_csv(out/'joint_resource_scenarios.csv',index=False)
    pd.DataFrame([r for rho in [-1,0,.5,1] for r in cost_moments(rho)]).to_csv(out/'cost_psa_evidence.csv',index=False)
    sub=[subgroup(kn,kf,p,sev,t,contrast=c) for kn in [2,3,4] for kf in [2,3,4]
         for p in [.25,401/1235,.5] for sev in [0,1] for t in [False,True]
         for c in ['equal_severe','hospital_risk_proxy','shared_contrast']]
    pd.DataFrame(sub).to_csv(out/'frequent_exacerbator_grid.csv',index=False)
    central=[subgroup(severe=sev,targeted=t,contrast=c) for sev in [0,1] for t in [False,True]
             for c in ['equal_severe','hospital_risk_proxy','shared_contrast']]
    pd.DataFrame(central).to_csv(out/'frequent_exacerbator_central.csv',index=False)
    summaries={}; draws={};prows=[]
    for src in ['tianjin','chongqing']:
        for maint in [112,500,1032]:
            for sev in [0,1]:
                key=f'{src}_maintenance_{maint}_severe_{sev}'
                s,dc,dq=psa(sev,maint,args.n,args.seed,cost_source=src)
                summaries[key]=s;draws[key]=(dc,dq)
                pd.DataFrame(dict(incremental_cost=dc,incremental_qaly=dq)).to_csv(out/f'psa_{key}.csv',index=False)
                prows.append(dict(scenario=key,mean_dc=s['dc'][0],low_dc=s['dc'][1],high_dc=s['dc'][2],mean_dq=s['dq'][0],low_dq=s['dq'][1],high_dq=s['dq'][2],p1=s['p1'],p2=s['p2'],evpi=s['evpi'],mcse_probability=s['mcse_probability']))
    pd.DataFrame(prows).to_csv(out/'psa_summary.csv',index=False)
    sens=[]
    for rho in [-1,0,.5,1]:
        for cv in [0,.1,.2,.3]:
            z,_,_=psa(1,n=args.n,seed=args.seed,cv=cv,rho=rho)
            sens.append(dict(rho=rho,transfer_cv=cv,mean_dc=z['dc'][0],low_dc=z['dc'][1],high_dc=z['dc'][2],p1=z['p1'],evpi=z['evpi']))
    pd.DataFrame(sens).to_csv(out/'cost_psa_sensitivity.csv',index=False)
    risk=[]
    for p0 in [*wilson(),12/18]:
        for sev in [0,1]:
            z,_,_=psa(sev,n=args.n,seed=args.seed,p0=p0)
            risk.append(dict(p0=p0,severe=sev,mean_dc=z['dc'][0],p1=z['p1'],evpi=z['evpi']))
    pd.DataFrame(risk).to_csv(out/'control_risk_conditional_psa.csv',index=False)
    mc=[]
    for n,seed in [(5000,args.seed),(args.n,args.seed),(args.n,args.seed+1),(args.n,args.seed+2)]:
        z,_,_=psa(1,n=n,seed=seed)
        mc.append(dict(n=n,seed=seed,p1=z['p1'],mean_dc=z['dc'][0],evpi=z['evpi'],mcse=z['mcse_probability']))
    pd.DataFrame(mc).to_csv(out/'monte_carlo_check.csv',index=False)
    ranges={'or_':(.10,.84),'p0':wilson(),'rm':(.16,.18),'rs':(.29,.33),
            'initial':(774,1290),'maint':(84,140),'m':(1-np.exp(-.04),1-np.exp(-.065)),
            'dm':(.0275,.0825),'ds':(.045,.135),'duration':(7,21),'h':(.25,1),'discount':(0,.05)}
    for k,r in zip(['cm','ch'],cost_moments()):ranges[k]=(BASE[k]*(1-2*r['total_cv']),BASE[k]*(1+2*r['total_cv']))
    ow=[dict(parameter=k,low=lo,high=hi,INMB_low=run(severe_effect=1,**{k:lo})['inmb'],INMB_high=run(severe_effect=1,**{k:hi})['inmb']) for k,(lo,hi) in ranges.items()]
    pd.DataFrame(ow).to_csv(out/'one_way_results.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=300)
        fig.savefig(figs/(name+'.tiff'),dpi=300,pil_kwargs={'compression':'tiff_lzw'})
        fig.savefig(figs/(name+'.pdf'))
        fig.savefig(figs/(name+'_1000dpi.tiff'),dpi=1000,pil_kwargs={'compression':'tiff_lzw'})
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.5),sharey=True)
    w=np.linspace(0,2*WTP,101)
    for ax,src,title in zip(axs,['tianjin','chongqing'],['A  Regional visit-cost proxies','B  Chongqing episode-cost scenario']):
        for sev,m,color,style,label in [(0,112,'#666666','--','Moderate only; maintenance 112'),(1,112,'#0072B2','-','Both severities; maintenance 112'),(1,500,'#E69F00','-','Both severities; maintenance 500'),(1,1032,'#009E73','-','Both severities; maintenance 1032')]:
            dc,dq=draws[f'{src}_maintenance_{m}_severe_{sev}']
            ax.plot(w/1000,[np.mean(x*dq-dc>0) for x in w],color=color,ls=style,label=label)
        ax.axvline(WTP/1000,color='#777777',lw=.7,ls=':');ax.set(title=title,xlabel='WTP (thousand CNY/QALY)',ylim=(-.02,1.02))
    axs[0].set_ylabel('Probability of cost-effectiveness')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=7)
    fig.tight_layout(rect=(0,.18,1,1));save(fig,'Figure1_CEAC')
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.4),sharex=True,sharey=True)
    for ax,src,title in zip(axs,['tianjin','chongqing'],['A  Regional visit-cost proxies','B  Chongqing episode-cost scenario']):
        dc,dq=draws[f'{src}_maintenance_112_severe_1']
        ax.scatter(dq[::10]*1000,dc[::10]/1000,s=3,alpha=.15,color='#0072B2',rasterized=True)
        detp=run(severe_effect=1,**(chongqing() if src=='chongqing' else {}))
        ax.scatter(detp['dq']*1000,detp['dc']/1000,marker='D',c='black',s=25,zorder=5)
        x=np.array([min(dq.min()*1000,-.25),max(dq.max()*1000,4)])
        ax.plot(x,WTP*x/1e6,ls=':',c='#D55E00',lw=1)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7)
        ax.set(title=title,xlabel='Incremental QALYs (× 10⁻³)')
    axs[0].set_ylabel('Incremental cost (thousand CNY)')
    fig.tight_layout();save(fig,'Figure2_CostEffectivenessPlane')
    b=run(severe_effect=1);factor=sum((1-BASE['m'])**t/(1+BASE['discount'])**t for t in [.5,1,2])
    cb=run(severe_effect=1,**chongqing())
    summary=dict(primary=run(),both=b,chongqing_primary=run(**chongqing()),chongqing_both=cb,
                 psa=summaries,base=BASE,wtp=WTP,p0_wilson=wilson(),cost_moments=cost_moments(),
                 maintenance_threshold=(b['maintenance']-b['dc'])/factor,
                 chongqing_maintenance_threshold=(cb['maintenance']-cb['dc'])/factor,
                 frequent=central,checks=check)
    def strict_json(value):
        if isinstance(value,dict):return {k:strict_json(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [strict_json(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
        return value
    (out/'summary.json').write_text(json.dumps(strict_json(summary),indent=2,allow_nan=False),encoding='utf-8')
    meta=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
              numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,n=args.n,seed=args.seed,
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=BASE,wtp=WTP,checks=check)
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['primary','both','chongqing_primary','chongqing_both','maintenance_threshold','chongqing_maintenance_threshold','checks']},indent=2))

if __name__=='__main__':main()
