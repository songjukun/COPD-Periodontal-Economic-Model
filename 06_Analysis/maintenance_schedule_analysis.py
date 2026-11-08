"""Reviewer 3 annual visit-frequency scenarios; hypothetical resources, not trial observations."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib
import numpy as np
import analysis as core
OUT=Path(__file__).resolve().parent/'results'
rows=[]
for source,cp in [('Tianjin',{}),('Chongqing',core.chongqing())]:
    for label,times in [('Once yearly',(1.,2.)),('Twice yearly',(.5,1.,1.5,2.))]:
        for cost in [200,500,1000]:
            for sev in [0,1]:
                r=core.run(maint=cost,maint_times=times,severe_effect=sev,**cp)
                ref=core.run(maint=cost,severe_effect=sev,**cp)
                expected=sum(cost*(1-core.BASE['m'])**t/(1+core.BASE['discount'])**t for t in times)
                assert np.isclose(r['maintenance'],expected)
                assert np.isclose(r['dc']-ref['dc'],r['maintenance']-ref['maintenance'])
                assert r['dq']==ref['dq']
                rows.append(dict(source=source,schedule=label,years=';'.join(map(str,times)),cost_per_visit=cost,severe_effect=sev,discounted_maintenance=float(r['maintenance']),dc=float(r['dc']),dq=float(r['dq']),inmb=float(r['inmb'])))
assert len(rows)==24
with (OUT/'maintenance_frequency_scenarios.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),rows=rows,checks=['24 schedule/price/source/severity scenarios','Maintenance reconciles to explicit survival/discount sum','Cost difference equals maintenance difference','Event benefit unchanged by assumed resource schedule'],core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitation='Schedules are resource-only scenarios; no observed attendance or effect-duration relationship is inferred.')
(OUT/'maintenance_frequency_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'checks':report['checks']},indent=2))
