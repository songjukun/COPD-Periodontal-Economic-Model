"""Revision 5 reporting analyses. No change to the clinical model or inputs.

Run analysis.py first, then this file. Outputs quantify reviewer-requested
cost-consequence ratios and the existing 7/14/21-day duration assumptions.
They are scenario calculations, not new clinical observations.
"""
from pathlib import Path
import json, hashlib, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import analysis as m

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def ratio(numerator, denominator):
    return float(numerator/denominator) if np.isfinite(denominator) and denominator>0 else None

def calculate():
    rows=[]; duration=[]
    for source in ('Tianjin','Chongqing'):
        costs={} if source=='Tianjin' else m.chongqing()
        for effect in (0,1):
            r=m.run(severe_effect=effect,**costs)
            events=r['am']+r['as_'];h=r['hospital_avoided']
            rows.append(dict(source=source,effect='Moderate only' if effect==0 else 'Both severities assumed',
                             incremental_cost=r['dc'],moderate_avoided=r['am'],severe_avoided=r['as_'],
                             all_events_avoided=events,hospitalizations_avoided=float(h) if np.isfinite(h) else None,
                             net_cost_per_event_avoided=ratio(r['dc'],events),
                             net_cost_per_hospitalization_avoided=ratio(r['dc'],h),
                             hospital_denominator_status='unidentifiable composite source' if source=='Chongqing'
                             else ('zero benefit; ratio undefined' if effect==0 else 'h=1 proxy'),
                             denominator_convention='undiscounted survival-adjusted events; discounted 2024 CNY costs'))
            for days in (7,14,21):
                x=m.run(severe_effect=effect,duration=days,**costs)
                assert np.isclose(x['dc'],r['dc'])
                assert np.isclose(x['dq'],r['dq']*days/14,rtol=1e-12)
                duration.append(dict(source=source,effect=rows[-1]['effect'],duration_days=days,
                                     incremental_cost=x['dc'],incremental_qaly=x['dq'],
                                     quality_adjusted_hours=x['dq']*365*24,inmb=x['inmb'],
                                     cost_effective_at_99665=bool(x['inmb']>0)))
    assert rows[0]['net_cost_per_hospitalization_avoided'] is None
    assert rows[2]['net_cost_per_hospitalization_avoided'] is None
    assert rows[3]['net_cost_per_hospitalization_avoided'] is None
    assert np.isclose(rows[1]['net_cost_per_event_avoided'],-8870.435579978588)
    for row in rows:
        assert np.isclose(row['all_events_avoided'],row['moderate_avoided']+row['severe_avoided'])
        assert np.isclose(row['net_cost_per_event_avoided']*row['all_events_avoided'],row['incremental_cost'])
    return rows,duration

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    rows,duration=calculate()
    pd.DataFrame(rows).to_csv(OUT/'event_cost_consequence.csv',index=False)
    pd.DataFrame(duration).to_csv(OUT/'duration_scenarios.csv',index=False)
    stored=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert len(stored['psa'])==12
    assert all(x['n']==50000 and x['seed']==20261015 for x in stored['psa'].values())
    data=dict(date_utc=datetime.now(timezone.utc).isoformat(),model_tests=m.tests(),
              event_rows=rows,duration_rows=duration,
              new_checks=['undefined zero/unknown hospital denominators retained as missing',
                          'event denominator reconciliation','cost ratio multiplication check',
                          'duration proportionality','duration does not change costs'],
              persistent_loss_scenario='Not estimated. ETHOS associations do not identify a Chinese periodontal-treatment causal benefit.',
              model_sha256=hashlib.sha256((HERE/'analysis.py').read_bytes()).hexdigest(),
              supplement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (OUT/'supplemental_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'event_rows':rows,'checks':data['new_checks'],'duration_range_primary':
                      [duration[0]['incremental_qaly'],duration[2]['incremental_qaly']]},indent=2))
