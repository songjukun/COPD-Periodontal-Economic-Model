"""Display retained model results; no parameter fitting or new PSA is performed."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
FIG=ROOT/'04_Figures'; RESULTS=ROOT/'06_Analysis/results'
for folder in ['PNG','300dpi_TIFF','Vector_PDF']:
    (FIG/folder).mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#0072B2'; ORANGE='#D55E00'; GREY='#5B6573'; TEAL='#00846B'
def load(name):
    with (RESULTS/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.savefig(FIG/'PNG'/(name+'.png'),dpi=300,facecolor='white')
    fig.savefig(FIG/'300dpi_TIFF'/(name+'.tiff'),dpi=300,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(FIG/'Vector_PDF'/(name+'.pdf'),facecolor='white')
    plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE,fill='#EFF6FA',fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',fc=fill,ec=color,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.3)
def arrow(ax,a,b,color=GREY,style='-',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,connectionstyle=f'arc3,rad={rad}',lw=1.3,color=color,linestyle=style))

# State transitions and event accrual are intentionally separated.
fig=plt.figure(figsize=(7.2,7.0))
ax=fig.add_axes([.04,.04,.92,.92]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(0,1,'A  Shared survival model and recurrent events',weight='bold',fontsize=11,va='top')
ax.text(.5,.948,'Usual care vs SRP + maintenance | two annual cycles',ha='center',fontsize=10)
box(ax,.03,.755,.38,.11,'ALIVE\nCOPD + periodontitis')
box(ax,.68,.755,.27,.11,'DEAD\nAbsorbing state',GREY,'#F4F5F6')
arrow(ax,(.42,.81),(.67,.81));ax.text(.54,.829,'m = 0.050386',ha='center',fontsize=9)
arrow(ax,(.13,.87),(.32,.87),rad=-.35);ax.text(.225,.922,'Remain alive: 1 − m',ha='center',fontsize=9)
ax.text(.81,.727,'No treatment effect on mortality',ha='center',fontsize=8.8)
arrow(ax,(.21,.747),(.21,.68),style='--')
box(ax,.03,.55,.42,.12,'Usual-care recurrent events\nModerate: 0.17/year; severe: 0.31/year',TEAL,'#F0F8F5',fontsize=9.2)
ax.text(.52,.62,'Events accrue while alive;\nnot mutually exclusive states.\nCosts + acute utility losses.',va='center',fontsize=9.3,linespacing=1.5)
ax.plot([0,1],[.515,.515],color='#D8DDE3',lw=1)
ax.text(0,.49,'B  Trial evidence and unvalidated effect mapping',weight='bold',fontsize=11,va='top')
box(ax,.025,.335,.27,.105,'Frequent-exacerbator\nOR 0.29; p₀ = 12/18',fontsize=9.4)
box(ax,.37,.335,.25,.105,'Approximate binary\nrisk ratio 0.550633',fontsize=9.4)
box(ax,.70,.315,.275,.145,'Assumed rate multiplier\nPrimary: moderate only\nAlternative: both severities',ORANGE,'#FFF5EF',fontsize=9.1)
arrow(ax,(.305,.39),(.36,.39));arrow(ax,(.63,.39),(.69,.39),ORANGE,'--')
ax.text(.49,.285,'No observed recurrent-event or severe-event treatment effect is supplied.',ha='center',fontsize=9,color=ORANGE)
ax.plot([0,1],[.255,.255],color='#D8DDE3',lw=1)
ax.text(0,.23,'C  Modeled dental resources and timing',weight='bold',fontsize=11,va='top')
ax.plot([.11,.91],[.12,.12],color=GREY,lw=1.5)
for x,month,label in [(.11,'0','SRP\nCNY 1,032'),(.31,'6','Maintenance'),(.51,'12','Maintenance'),(.91,'24','Maintenance')]:
    ax.plot(x,.12,'o',color=BLUE,ms=5)
    ax.text(x,.14,label,ha='center',va='bottom',fontsize=9)
    ax.text(x,.095,month,ha='center',va='top',fontsize=9)
ax.text(.68,.082,'Months',ha='center',fontsize=9)
ax.text(.5,.02,'Maintenance floor: CNY 112/visit; survival-weighted and discounted.\nResources and efficacy under alternative schedules remain assumptions.',ha='center',fontsize=8.8)
save(fig,'Figure1_Model_Structure')

data={r['scenario']:r for r in load('scenario_results.csv')}
selected=[('No COPD effect','No COPD effect'),('Primary moderate only','Primary: moderate only'),('Severe reduction attenuated 50%','Severe reduction attenuated 50%'),('Equal effects on both severities','Equal reduction at both severities'),('First-year-only effect; moderate only','First year only: moderate'),('First-year-only effect; both severities','First year only: both severities'),('Upper OR moderate only','OR 0.84: moderate only'),('Upper OR both severities','OR 0.84: both severities')]
fig,(a,b)=plt.subplots(1,2,figsize=(8.8,5.8),sharey=True,gridspec_kw={'width_ratios':[1.5,1]})
fig.subplots_adjust(left=.34,right=.97,top=.84,bottom=.18,wspace=.34)
for i,(key,label) in enumerate(selected):
    r=data[key];dc=float(r['dc']);dq=float(r['dq'])*1000
    c=BLUE if i==1 else GREY
    a.hlines(i,0,dc,color=c,lw=2);a.plot(dc,i,'o',color=c,ms=5)
    a.annotate(f'{dc:,.0f}',(dc,i),xytext=((5 if dc>=0 else -5),0),textcoords='offset points',ha='left' if dc>=0 else 'right',va='center',fontsize=8.5)
    b.hlines(i,0,dq,color=c,lw=2);b.plot(dq,i,'o',color=c,ms=5)
    b.annotate(f'{dq:.3f}',(dq,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8.5)
a.set_yticks(range(len(selected)),[label for _,label in selected],fontsize=9.2)
a.invert_yaxis();a.set_xlim(-4700,2150);b.set_xlim(-.04,1.48)
for ax in [a,b]:ax.axvline(0,color='#111111',lw=.8);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
a.set_xlabel('Incremental cost (2024 CNY/patient)');b.set_xlabel('Incremental QALYs (× 10⁻³)')
a.set_title('A  Cost',loc='left',weight='bold');b.set_title('B  Health gain',loc='left',weight='bold')
fig.suptitle('Selected structural scenarios, not clinical effect estimates',fontsize=12,weight='bold',y=.965)
fig.text(.5,.90,'Tianjin costs; h = 1; maintenance CNY 112 at 6/12/24 months',ha='center',fontsize=9.5)
fig.text(.5,.05,'Negative cost denotes savings. Points are deterministic; no confidence intervals are implied.\nThe primary scenario is highlighted. Complete numerical scenarios are retained in Table S3.',ha='center',fontsize=9)
save(fig,'FigureS1_Structural_Scenarios')

labels={'or_':'Trial odds ratio','p0':'Control binary risk','rm':'Moderate-event rate','rs':'Severe-event rate','initial':'Initial SRP cost','maint':'Maintenance cost/visit','m':'Annual mortality probability','dm':'Moderate-event disutility','ds':'Severe-event disutility','duration':'Acute disutility duration','h':'Hospital share of severe events','discount':'Annual discount rate','cm':'Outpatient cost proxy','ch':'Hospital cost proxy'}
ow=sorted(load('one_way_results.csv'),key=lambda r:abs(float(r['INMB_high'])-float(r['INMB_low'])),reverse=True)
center=float(data['Equal effects on both severities']['inmb'])
fig,ax=plt.subplots(figsize=(8.5,6.4));fig.subplots_adjust(left=.38,right=.97,top=.80,bottom=.16)
for i,r in enumerate(ow):
    for key,c in [('INMB_low',BLUE),('INMB_high',ORANGE)]:
        value=float(r[key]);ax.barh(i,value-center,left=center,height=.64,color=c,alpha=.9)
ax.set_yticks(range(len(ow)),[labels[r['parameter']] for r in ow],fontsize=9.2);ax.invert_yaxis()
ax.axvline(center,color='#222222',lw=1.1,label='Central INMB');ax.axvline(0,color='#777777',ls='--',lw=1)
ax.set_xlim(-1000,7600);ax.set_xlabel('Incremental net monetary benefit (2024 CNY/patient)')
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('One-way sensitivity within an assumed both-severity scenario',fontsize=11.5,weight='bold',y=.96)
fig.text(.5,.89,'Tianjin; maintenance CNY 112/visit; WTP CNY 99,665/QALY\nCentral INMB = CNY 3,752.73; not the moderate-only primary scenario',ha='center',fontsize=9.6)
fig.legend([Line2D([0],[0],lw=7,color=BLUE),Line2D([0],[0],lw=7,color=ORANGE),Line2D([0],[0],lw=1,color='#222222')],['Parameter at lower bound','Parameter at upper bound','Central INMB'],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01),fontsize=8.8)
save(fig,'FigureS2_OneWay_Sensitivity')
record={'scope':'Visualization only; no model inputs or retained numerical outputs changed.',
        'model_schematic_sources':['Methods model structure, timing and treatment-effect mapping','analysis.py'],
        'scenario_plot':[data[k] for k,_ in selected], 'one_way_plot':ow,
        'files':{n:hashlib.sha256((RESULTS/n).read_bytes()).hexdigest() for n in ['scenario_results.csv','one_way_results.csv']},
        'dpi':{}}
for p in (FIG/'300dpi_TIFF').glob('*.tiff'):
    with Image.open(p) as im:
        assert tuple(map(float,im.info['dpi']))==(300.,300.)
        record['dpi'][p.name]={'pixels':im.size,'dpi':list(map(float,im.info['dpi']))}
(RESULTS/'figure_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Generated 3 additional figures; {len(record["dpi"])} TIFFs verified as 300 dpi.')
