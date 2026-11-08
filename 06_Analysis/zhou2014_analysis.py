"""Deterministic, explicitly hypothetical time-specific mapping; no new trial fit.
Run after analysis.py/supplemental_analysis.py, or use their supplied archived outputs.
No joint PSA is invented for correlated repeated-year adjusted ORs.
"""
from pathlib import Path
import json,hashlib,math,csv
from datetime import datetime,timezone
import numpy as np
import analysis as core

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'07_Evidence/Zhou2014_verified_inputs.json'
E=json.loads(INPUT.read_text(encoding='utf-8'))

def mapped(odds,p0):
    assert odds>0 and 0<p0<1
    return odds/(1-p0+p0*odds)

def temporal(p,odds,risks):
    assert len(odds)==len(risks)==2
    rr=[mapped(o,p0) for o,p0 in zip(odds,risks)]
    cs=p['ch'] if p['composite'] else p['h']*p['ch']+(1-p['h'])*p['ed']
    uc=events_cost=dq=am=asev=0.
    for y in range(2):
        ex=(1-p['m'])**y*(1-p['m']/2)
        de=ex/(1+p['discount'])**(y+.5)
        reduction=(1-rr[y])*p['share']*(0 if p['first_only'] and y==1 else 1)
        uc+=de*(p['rm']*p['cm']+p['rs']*cs)
        events_cost+=de*(p['rm']*(1-reduction)*p['cm']+p['rs']*(1-reduction*p['severe_effect'])*cs)
        dq+=de*reduction*(p['rm']*p['dm']+p['rs']*p['ds']*p['severe_effect'])*p['duration']/365
        am+=ex*p['rm']*reduction
        asev+=ex*p['rs']*reduction*p['severe_effect']
    maint=sum(p['maint']*(1-p['m'])**t/(1+p['discount'])**t for t in [.5,1,2])
    tr=p['initial']+maint+events_cost
    dc=tr-uc
    return dict(dc=dc,dq=dq,inmb=core.WTP*dq-dc,am=am,as_=asev,
                rr_year1=rr[0],rr_year2=rr[1],initial=p['initial'],maintenance=maint,
                hospital_avoided=None if p['composite'] else p['h']*asev)

def main():
    checks=[]
    core_tests=core.tests()
    assert all(r['frequent']+r['infrequent']==r['observed_n'] for r in E['table4_counts'])
    checks.append('All nine observed denominators reconcile')
    for r in E['table5']:assert 0<r['lower']<r['or']<r['upper'] and 0<r['p']<1
    checks.append('All six published adjusted OR/CI/P rows transcribed and bounded')
    for source,cp in [('Tianjin',{}),('Chongqing',core.chongqing())]:
        for sev in [0,1]:
            for odds in [.04,.29,1.]:
                p={**core.BASE,**cp,'severe_effect':sev,'or_':odds}
                old=core.model(p);new=temporal(p,[odds]*2,[p['p0']]*2)
                for k in ['dc','dq','inmb','am','as_','initial','maintenance']:
                    assert math.isclose(old[k],new[k],rel_tol=1e-10,abs_tol=1e-10),(k,old[k],new[k])
    checks.append('Constant-year kernel reproduces original across 12 cost/severity/effect cases, including null')
    rows=[]
    for source,cp in [('Tianjin',{}),('Chongqing',core.chongqing())]:
        for arm,initial in [('SRP',1032.),('Scaling',112.)]:
            odds=[r['or'] for r in E['table5'] if r['comparison']==arm+' vs Control']
            for mapping in ['Year2 constant','Year-specific']:
                os=odds if mapping=='Year-specific' else [odds[1]]*2
                ps=E['control_risks'] if mapping=='Year-specific' else [E['control_risks'][1]]*2
                for sev in [0,1]:
                    p={**core.BASE,**cp,'initial':initial,'severe_effect':sev}
                    r=temporal(p,os,ps)
                    rows.append(dict(source=source,arm=arm,mapping=mapping,severe_effect=sev,
                                     or_year1=os[0],or_year2=os[1],p0_year1=ps[0],p0_year2=ps[1],**r))
    assert len(rows)==16
    assert all(r['as_']==0 for r in rows if r['severe_effect']==0)
    assert all(r['hospital_avoided'] is None for r in rows if r['source']=='Chongqing')
    checks.append('No invented hospitalization denominator; no severe benefit in moderate-only scenarios')
    old=json.loads((ROOT/'06_Analysis/results/summary.json').read_text(encoding='utf-8'))
    ref=next(r for r in rows if (r['source'],r['arm'],r['mapping'],r['severe_effect'])==('Tianjin','SRP','Year2 constant',0))
    assert math.isclose(ref['dc'],old['primary']['dc'],abs_tol=1e-9)
    assert math.isclose(ref['dq'],old['primary']['dq'],abs_tol=1e-12)
    checks.append('Archived primary result reproduced; main inputs unchanged')
    path=ROOT/'06_Analysis/results/zhou2014_time_scenarios.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    source_path=Path(E['source_path']) if E.get('source_path') else None
    source_present=source_path is not None and source_path.is_file()
    if source_present:
        assert sha(source_path)==E['verified_pdf_sha256'],'Source file differs from the verified primary PDF'
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),checks=checks,rows=rows,
                core_tests=core_tests,
                script_sha256=sha(Path(__file__)),input_sha256=sha(INPUT),core_sha256=sha(ROOT/'06_Analysis/analysis.py'),
                source_pdf_sha256=E['verified_pdf_sha256'],source_pdf_hash_rechecked_this_run=source_present,
                main_PSA='See results/run_metadata.json for the actual core run. No joint time-specific PSA is fitted.',
                limitations='Adjusted OR plus marginal control risk remains an approximate binary-risk mapping, not a recurrent-event rate ratio. Correlated year-specific uncertainty is unknown; no joint year-specific PSA, efficacy ranking, or time-interaction inference.')
    (ROOT/'06_Analysis/results/zhou2014_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
