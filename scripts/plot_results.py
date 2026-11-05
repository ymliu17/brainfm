"""Replot current main figures and supplementary result panels from aggregate data.
No participant data, model fitting, inference recomputation or manuscript tools.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/brainfm_plot_cache')
from pathlib import Path
import argparse,json,hashlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.patches import Patch,Rectangle
import plot_results_base as base
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures'
INPUT_ROOT=Path(os.environ.get('BRAINFM_RESULTS_DIR',str(ROOT/'results'))).expanduser().resolve()
DATA=INPUT_ROOT/'figure_data'
TABLES=INPUT_ROOT/'tables'
METHODS=base.METHODS;FM=METHODS[2:];C=base.COLORS;LABEL=base.SHORT;NAMES=base.NAMES
COS=['adni','nacc','act','cogte','breathe','psopt','hrv']
ROWS=FM+['braingfm','sf100_fc']
RCT_COHORTS=['act','cogte','breathe','psopt','hrv']
style=SimpleNamespace(base=base)
HSTEM='Figure2_performance_heatmap'
CHECKS={};CAP={};INPUTS={}
STEMS={4:'Figure3_cognition_intervention',5:'Figure4_posttraining_alignment'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
def read(name,controls=False):
    p=DATA/(('Identity_controls_' if controls else '')+name+'.csv')
    INPUTS[str(p.relative_to(INPUT_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,low_memory=False)
def table(dataset,sheet):
    p=TABLES/f'Dataset_S{dataset}__{sheet}.csv'
    INPUTS[str(p.relative_to(INPUT_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,low_memory=False)
def write(fig,stem):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/(stem+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.1,facecolor='white')
    plt.close(fig)

def title(ax, letter, label):
    ax.set_title(r'$\mathbf{'+letter+'}$  '+label, loc='left', fontsize=9, pad=11)

def clean(ax, xlabel=None):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=7, length=2)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)

def zero(ax, xlabel):
    ax.axvline(0, color='#B5BBC1', ls=':', lw=.7, zorder=0)
    clean(ax, xlabel)

def interval(ax, r, y, color, marker='o', ls='-', size=15):
    lo,hi = json.loads(r.ci95) if isinstance(r.get('ci95'), str) else (r.ci_low, r.ci_high)
    assert np.isfinite([lo,hi,r.delta,r.holm_p]).all() and lo <= hi
    ax.plot([lo,hi],[y,y],color=color,lw=.85,ls=ls,alpha=.85)
    ax.scatter(r.delta,y,s=size,marker=marker,color=color,zorder=3)
    if r.holm_p < .05:
        ax.scatter(r.delta,y,s=39,marker='*',color=color,edgecolor='#202020',linewidth=.3,zorder=4)

def legend(fig, handles, ncol=4, y=.015, gap_points=None):
    # Matplotlib fills columns: transpose to retain intuitive row order.
    handles=[handles[i] for j in range(ncol) for i in range(j,len(handles),ncol)]
    loc = 'lower center'
    if gap_points is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lowest = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
        y = (lowest - gap_points * fig.dpi / 72) / fig.bbox.height
        loc = 'upper center'
    fig.legend(handles=handles,loc=loc,bbox_to_anchor=(.52,y),ncol=ncol,
               frameon=False,fontsize=7,columnspacing=1.2,handlelength=2.3,labelspacing=.8,
               **({'borderaxespad':0} if gap_points is not None else {}))

def models(ax):
    ax.set_yticks(range(9),[LABEL[k] for k in FM],fontsize=7)
    ax.set_ylim(8.6,-.6)

def clinical(ax,q,ep,kind):
    groups=[]
    for co in ['act','cogte','breathe','psopt']+(['hrv'] if ep=='EF' else []):
        g=q[q.cohort.eq(co)&q.task.eq(ep+'_'+kind)]
        if co=='hrv' and kind=='followup':
            for val in sorted(g.reference_score.round(6).unique()):
                groups.append(('HRV-ER'+('†' if val<.8 else '‡'),g[np.isclose(g.reference_score,val,atol=1e-6)]))
        else:groups.append((NAMES[co],g))
    for i,(_,g) in enumerate(groups):
        for _,r in g.iterrows():
            interval(ax,r,i+(METHODS.index(r.model)-5)*.060,C[r.model],size=8)
    ax.set_yticks(range(len(groups)),[n for n,_ in groups]);ax.set_ylim(len(groups)-.5,-.5)
    zero(ax,'Δρ: clinical + imaging − clinical only')

def figure4():
    r=read('Figure5A_cognitive_readouts');f=read('Figure4C_baseline_imaging_followup_increment');c=read('Figure4D_longitudinal_imaging_change_increment')
    fig,axs=plt.subplots(3,2,figsize=(8.2,10.1));fig.subplots_adjust(left=.16,right=.98,top=.96,bottom=.16,hspace=.47,wspace=.60)
    for j,ep in enumerate(['MEM','EF']):
        ax=axs[0,j]
        for i,model in enumerate(FM):
            for k,(co,rd) in enumerate((co,rd) for co in ['adni','nacc'] for rd in ['spatial_flat','connectivity']):
                z=r[r.model.eq(model)&r.cohort.eq(co)&r.task.eq(ep)&r.candidate.eq(rd)]
                assert len(z)==1
                interval(ax,z.iloc[0],i+(k-1.5)*.17,C[model],marker='o' if rd=='spatial_flat' else 's',ls='-' if co=='adni' else '--',size=10)
        models(ax);title(ax,'AB'[j],('Memory' if ep=='MEM' else 'EF')+': readout effects');zero(ax,'Δρ: alternative − primary readout')
        clinical(axs[1,j],f,ep,'followup');title(axs[1,j],'CD'[j],('Memory' if ep=='MEM' else 'EF')+': follow-up increment')
        clinical(axs[2,j],c,ep,'change');title(axs[2,j],'EF'[j],('Memory' if ep=='MEM' else 'EF')+': change increment')
    for row in axs:
        lim=(min(a.get_xlim()[0] for a in row),max(a.get_xlim()[1] for a in row))
        for ax in row:ax.set_xlim(lim)
    h=[Line2D([],[],color=C[k],marker='o',label=LABEL[k]) for k in METHODS]
    h += [Line2D([],[],color='#444',ls=ls,label='A–B: '+n) for ls,n in [('-','ADNI'),('--','NACC')]]
    h += [Line2D([],[],color='#444',ls='',marker=mk,label='A–B: '+n) for mk,n in [('o','Flattened spatial'),('s','Connectivity')]]
    legend(fig,h,5,gap_points=9)
    save(fig,4)
    CHECKS['Figure4']={'readout_comparisons':len(r),'followup_increment':len(f),'change_increment':len(c)}

def figure5():
    s=read('Figure5C_within_source_posttraining');e=read('Figure5D_external_posttraining');a=read('FigureS3_all_alignment')
    fig,axs=plt.subplots(3,2,figsize=(8.2,9.1));fig.subplots_adjust(left=.16,right=.98,top=.95,bottom=.09,hspace=.56,wspace=.60)
    for j,co in enumerate(['adni','nacc']):
        ax=axs[0,j]
        for i,model in enumerate(FM):
            for k,ep in enumerate(['MEM','EF']):
                z=s[s.model.eq(model)&s.cohort.eq(co)&s.task.eq(ep)]
                assert len(z)==1
                interval(ax,z.iloc[0],i+(k-.5)*.27,C[model],marker=['o','s'][k])
        models(ax);title(ax,'AB'[j],NAMES[co]+': within-source');zero(ax,'Δρ: post-trained − frozen')
        ax=axs[1,j]
        z=e[e.display_setting.eq(['fixed_source_transfer','local_target_probe'][j])]
        for i,model in enumerate(FM):
            q=z[z.model.eq(model)];assert len(q)==1
            interval(ax,q.iloc[0],i,C[model])
        models(ax);title(ax,'CD'[j],['ADNI → NACC: memory','BREATHE: baseline memory'][j]);zero(ax,'Δρ: post-trained − frozen')
        ax=axs[2,j];z=a[a.setting.eq(['source_axis','local_followup'][j])]
        for i,model in enumerate(FM):
            q=z[z.model.eq(model)].sort_values(['source','cohort','task','objective','candidate'],kind='stable')
            # Fixed within-model spread reveals all comparisons; it is not uncertainty.
            yy=i+np.linspace(-.31,.31,len(q))
            ax.scatter(q.delta,yy,s=7,color=C[model],alpha=.32,linewidths=0)
            sig=q.holm_p.to_numpy()<.05
            ax.scatter(q.delta.to_numpy()[sig],yy[sig],s=43,marker='*',color=C[model],edgecolor='#222',linewidth=.35,zorder=4)
        models(ax);title(ax,'EF'[j],['Alignment: source transfer','Alignment: local follow-up'][j]);zero(ax,'Δρ: aligned − unaligned')
    for row in axs:
        lim=(min(ax.get_xlim()[0] for ax in row),max(ax.get_xlim()[1] for ax in row))
        for ax in row:ax.set_xlim(lim)
    handles=[Line2D([],[],color=C[k],marker='o',label=LABEL[k]) for k in FM]
    handles += [Line2D([],[],marker=mk,ls='',color='#444',label='A–B: '+name) for mk,name in [('o','Memory'),('s','EF')]]
    legend(fig,handles,6,gap_points=9)
    save(fig,5)
    CHECKS['Figure5']={'within_source':len(s),'external_selected_settings':len(e),'all_alignment_comparisons':len(a)}

def plot_heatmap(data):
    cols=list(data[['cohort','task']].drop_duplicates().itertuples(index=False,name=None))
    mat=np.array([[data[(data.model==m)&(data.cohort==co)&(data.task==t)].delta_fc.iloc[0] for co,t in cols] for m in ROWS])
    bound=max(.1,np.ceil(np.nanmax(abs(mat))*10)/10)
    cmap=LinearSegmentedColormap.from_list('FC_difference',['#2166ac','#ffffff','#b35806']);cmap.set_bad('#dedede')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.35})
    fig,ax=plt.subplots(figsize=(11.0,5.55));fig.subplots_adjust(left=.173,right=.99,top=.85,bottom=.32)
    im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto',interpolation='nearest')
    labels=['BrainLM 13M','BrainLM 111M','BrainLM 650M','Brain-JEPA','Brain-Semantoks','fMRI-LM VQ-contrastive','fMRI-LM MAE','fMRI-LM VQ-domain','BrainHarmonix-F','BrainGFM','Schaefer100 FC']
    ax.set_yticks(range(11),labels,fontsize=8)
    names={'AGE':'DEMO (age)','AGE_GROUP':'DEMO (age cat.)','SEX':'DEMO (sex)','DX':'DX','MEM':'COG (MEM)','EF':'COG (EF)','MEM_followup':'COG-FU (MEM)','EF_followup':'COG-FU (EF)','MEM_change':'ΔCOG (MEM)','EF_change':'ΔCOG (EF)','arm':'ARM','identification':'Re-ID'}
    ax.set_xticks(range(len(cols)),[names[t] for co,t in cols],rotation=55,ha='right',rotation_mode='anchor',fontsize=7.2)
    ax.tick_params(length=0,pad=4)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True);ax.set_yticks(np.arange(-.5,11,1),minor=True)
    ax.grid(which='minor',color='white',lw=.35);ax.tick_params(which='minor',length=0)
    for i,m in enumerate(ROWS):
        for j,(co,t) in enumerate(cols):
            r=data[(data.model==m)&(data.cohort==co)&(data.task==t)].iloc[0]
            if not r.available:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='///',edgecolor='#8b8b8b',linewidth=0))
            elif m=='sf100_fc':ax.text(j,i,f'{r.score:.2f}',ha='center',va='center',rotation=90,fontsize=6.5,color='#242424',zorder=4)
    ax.axhline(8.5,color='#444444',lw=.9)
    for k,co in enumerate(COS):
        indices=[j for j,(c,t) in enumerate(cols) if c==co];lo=min(indices);hi=max(indices)
        if k:ax.axvline(lo-.5,color='white',lw=2.5)
        ax.text((lo+hi)/2,1.04,style.base.NAMES[co],transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,fontweight='bold')
    for lo,hi,label in [(0,11,'Observational cohorts'),(12,len(cols)-1,'Intervention cohorts')]:
        ax.plot([lo-.15,lo-.15,hi+.15,hi+.15],[1.11,1.145,1.145,1.11],transform=ax.get_xaxis_transform(),clip_on=False,color='#626262',lw=.85,solid_capstyle='butt',solid_joinstyle='miter')
        ax.text((lo+hi)/2,1.18,label,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9)
    ax.spines[:].set_visible(False)
    cax=fig.add_axes([.25,.165,.46,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_ticks([-bound,-bound/2,0,bound/2,bound]);cb.ax.tick_params(labelsize=8,length=2)
    cb.set_label('Performance difference from Schaefer100 FC',fontsize=8)
    handles=[Patch(facecolor='#dedede',hatch='///',edgecolor='#8b8b8b',label='Outcome unavailable')]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.745,.171),ncol=1,frameon=False,fontsize=8,borderaxespad=0)
    write(fig,HSTEM)

_PENDING={}
_original_legend=legend
def legend(fig,handles,ncol=4,**kwargs):
    _PENDING[id(fig)]=(handles,ncol)
def save(fig,num):
    fig.set_size_inches(8.8,10.5 if num==4 else 10.0)
    fig.subplots_adjust(left=.16,right=.98,top=.95,bottom=.09,hspace=.85,wspace=.62)
    labels={4:[('COG (MEM): Readout','Δρ (alternative − primary readout)'),('COG (EF): Readout','Δρ (alternative − primary readout)'),('COG-FU (MEM)','Δρ (clinical + imaging − clinical only)'),('COG-FU (EF)','Δρ (clinical + imaging − clinical only)'),('ΔCOG (MEM)','Δρ (clinical + imaging − clinical only)'),('ΔCOG (EF)','Δρ (clinical + imaging − clinical only)')],5:[('COG: ADNI post-training','Δρ (post-trained − frozen)'),('COG: NACC post-training','Δρ (post-trained − frozen)'),('COG (MEM): ADNI → NACC','Δρ (post-trained − frozen)'),('COG (MEM): BREATHE','Δρ (post-trained − frozen)'),('Cognitive transfer: Alignment','Δρ (aligned − unaligned)'),('COG-FU: Alignment','Δρ (aligned − unaligned)')]}
    for i,(ax,(label,xlabel)) in enumerate(zip(fig.axes,labels[num])):
        title(ax,chr(65+i),label);ax.set_xlabel(xlabel,fontsize=7.2)
    fig.canvas.draw()
    for row in [1,2]:
        renderer=fig.canvas.get_renderer()
        upper=[a.get_tightbbox(renderer) for a in fig.axes[2*(row-1):2*row]]
        lower=[a.get_tightbbox(renderer) for a in fig.axes[2*row:2*row+2]]
        shift=(min(b.y0 for b in upper)-max(b.y1 for b in lower)-12*fig.dpi/72)/fig.bbox.height
        for ax in fig.axes[2*row:]:
            pos=ax.get_position();ax.set_position([pos.x0,pos.y0+shift,pos.width,pos.height])
        fig.canvas.draw()
    handles,ncol=_PENDING.pop(id(fig));_original_legend(fig,handles,ncol,gap_points=9)
    write(fig,STEMS[num])

def cohort_profiles():
    data=table(15,'S1_relocated_radar')
    tasks={c:['AGE','SEX','DX','MEM','EF','identification'] if c in ('adni','nacc') else ['AGE','SEX','MEM','EF','MEM_change','EF_change','arm','identification'] for c in COS}
    tasks['hrv']=['SEX','MEM_delayed_recall','EF','EF_change','arm','identification']
    fig,axs=plt.subplots(4,2,subplot_kw={'projection':'polar'},figsize=(9,13))
    fig.subplots_adjust(hspace=.65,wspace=.5,bottom=.06,top=.96)
    for ax,co in zip(axs.flat,COS):
        ts=tasks[co];angle=np.linspace(0,2*np.pi,len(ts),endpoint=False)
        for model in METHODS:
            q=data[data.cohort.eq(co)&data.method.eq(model)].set_index('task').reindex(ts)
            v=q.score.to_numpy(float)
            # Radar display maps Spearman [-1,1] to [0,1]; classification/Re-ID retain their scales.
            for j,t in enumerate(ts):
                if t not in ('SEX','DX','arm','identification'):v[j]=(v[j]+1)/2
            ax.plot(np.r_[angle,angle[0]],np.r_[v,v[0]],color=C[model],lw=.9,marker='o',ms=2,ls='--' if model=='sf100_fc' else '-')
        ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,1)
        ax.set_xticks(angle,[t.replace('identification','Re-ID').replace('_','\n') for t in ts],fontsize=7)
        ax.set_title(NAMES[co],pad=24);ax.set_yticklabels([])
    axs.flat[-1].set_axis_off()
    axs.flat[-1].legend(handles=[Line2D([],[],color=C[m],label=base.LABEL[m],ls='--' if m=='sf100_fc' else '-') for m in METHODS],loc='center',frameon=False,fontsize=8)
    write(fig,'FigureS1_cohort_profiles')

def cloud(ax,q,label,xlabel='Performance difference'):
    for i,model in enumerate(FM):
        g=q[q.model.eq(model)]
        if not len(g):continue
        y=i+np.linspace(-.27,.27,len(g))
        ax.scatter(g.delta,y,s=9,color=C[model],alpha=.4)
        p=g['p_adjusted'] if 'p_adjusted' in g else g.holm_p
        sig=(p<.05).to_numpy()
        ax.scatter(g.delta.to_numpy()[sig],y[sig],s=35,marker='*',color=C[model],edgecolor='#222',linewidth=.3)
    models(ax);zero(ax,xlabel);ax.set_title(label,loc='left',fontsize=9)

def grid_cloud(stem,panels,ncol=2):
    nr=(len(panels)+ncol-1)//ncol
    fig,axs=plt.subplots(nr,ncol,figsize=(5.3*ncol,3.9*nr),squeeze=False)
    fig.subplots_adjust(left=.15,right=.98,bottom=.09,top=.94,hspace=.55,wspace=.65)
    for i,(q,label,xlabel) in enumerate(panels):cloud(axs.flat[i],q,chr(65+i)+'  '+label,xlabel)
    for ax in list(axs.flat)[len(panels):]:ax.set_axis_off()
    write(fig,stem)

def controls_figure():
    scores=read('Figure3_AB_scores');paired=read('Figure3_CDF_paired_tests');counts=read('Figure3_E_counts')
    fig,axs=plt.subplots(3,2,figsize=(9,11));fig.subplots_adjust(left=.16,right=.98,top=.96,bottom=.13,hspace=.6,wspace=.58)
    cohorts=['adni','nacc_repeated','act','cogte','breathe','psopt','hrv']
    symbols={'pretrained_native':('CLS',C['brainlm13'],'o',':'),'pretrained_connectivity':('Connectivity',C['brainlm13'],'s','-'),'regional_norm':('Regional norms',C['brainlm13'],'^','--'),'random_connectivity':('Random connectivity',C['brainlm13'],'x','-.'),'FC':('Input-window FC','#aaa','D','-'),'roi_std':('Regional BOLD SD','#444','v','--')}
    for j,panel in enumerate('AB'):
        ax=axs[0,j];q=scores[scores.panel.eq(panel)]
        for method,(name,color,marker,ls) in symbols.items():
            v=q[q.method.eq(method)].set_index('cohort').loc[cohorts,'score']
            ax.plot(range(7),v,color=color,marker=marker,ls=ls,ms=4,lw=1,mfc='white',label=name)
        ax.set_xticks(range(7),[NAMES[c] for c in cohorts],rotation=35,ha='right');ax.set_ylim(-.025,1.025);ax.set_ylabel('Re-ID accuracy');title(ax,panel,['Native dimensions','Matched 128 dimensions'][j]);clean(ax)
    for panel,ax in [('C',axs[1,0]),('D',axs[1,1]),('F',axs[2,1])]:
        q=paired[paired.figure_panel.eq(panel)];keys=['cohort'] if panel!='F' else ['comparison']
        labels=list(q[keys[0]].drop_duplicates())
        for i,key in enumerate(labels):
            for j,mode in enumerate(['native','projected128']):
                g=q[q[keys[0]].eq(key)&q['mode'].eq(mode)]
                if len(g):interval(ax,g.iloc[0],i+(j-.5)*.25,C['brainlm13'],marker=['o','s'][j])
        ax.set_yticks(range(len(labels)),[NAMES.get(k,k.replace('split','Split ').replace('_connectivity_minus_',' vs. ')) for k in labels],fontsize=7);ax.set_ylim(len(labels)-.5,-.5)
        title(ax,panel,{'C':'Pretraining × readout','D':'Temporal-order sensitivity','F':'pSOPT: FD-adjusted'}[panel]);zero(ax,'Δ Re-ID accuracy' if panel!='C' else 'Difference in readout gains')
    ax=axs[2,0]
    # Exported counts retain the original complete nine-configuration correction families.
    count_col=next(c for c in counts if c in ['significant_advantages','significant_cohorts','n_cohorts','count','n_significant'])
    for i,model in enumerate(FM):
        for j,mode in enumerate(['native','projected128']):
            q=counts[counts.model.eq(model)&counts['mode'].eq(mode)]
            ax.scatter(q[count_col],np.full(len(q),i+(j-.5)*.25),color=C[model],marker=['o','s'][j])
    models(ax);ax.set_xlim(-.3,7.3);ax.set_xticks(range(8));ax.set_xlabel('Cohorts with corrected pretraining advantage');title(ax,'E','Pretrained versus random')
    handles=[Line2D([],[],color=co,marker=mk,ls=ls,label=name) for name,co,mk,ls in symbols.values()]
    handles += [Line2D([],[],color='#444',marker=mk,ls='',label=label) for mk,label in [('o','Native'),('s','128 dimensions')]]
    _original_legend(fig,handles,4,gap_points=10);write(fig,'FigureS3_readout_identification')

def performance_grid(data,stem,keys,value='score'):
    fig,axs=plt.subplots(1,len(keys),figsize=(5*len(keys),5),squeeze=False)
    fig.subplots_adjust(left=.14,right=.95,bottom=.27,top=.86,wspace=.65)
    for ax,(q,label) in zip(axs.flat,keys):
        cols=list(q.cohort.drop_duplicates());rows=[m for m in METHODS if m in q.model.unique()]
        a=q.pivot_table(index='model',columns='cohort',values=value,aggfunc='first').reindex(index=rows,columns=cols)
        im=ax.imshow(a,vmin=-1 if value!='top1' else 0,vmax=1,cmap='RdBu_r',aspect='auto')
        ax.set_yticks(range(len(rows)),[base.LABEL[m] for m in rows],fontsize=7);ax.set_xticks(range(len(cols)),[NAMES.get(c,c) for c in cols],rotation=45,ha='right',fontsize=7);ax.set_title(label)
    fig.colorbar(im,ax=axs.ravel().tolist(),orientation='horizontal',fraction=.04,pad=.24,label=value)
    write(fig,stem)

def supplementary():
    cohort_profiles();controls_figure()
    a=table(12,'S12_cohort_evidence_tests')
    # Absolute tests are reported as adjusted P values, never inferred from paired differences.
    a.to_csv(OUT/'absolute_evidence.csv',index=False)
    t=table(15,'S1_absolute_performance');tasks=list(t.task.drop_duplicates())
    fig,axs=plt.subplots((len(tasks)+2)//3,3,figsize=(12,3.3*((len(tasks)+2)//3)),squeeze=False)
    fig.subplots_adjust(left=.12,right=.98,bottom=.08,top=.95,hspace=.8,wspace=.6)
    for ax,task in zip(axs.flat,tasks):
        g=t[t.task.eq(task)];co=list(g.cohort.drop_duplicates());a=g.pivot_table(index='model',columns='cohort',values='p_adjusted',aggfunc='first').reindex(index=FM,columns=co)
        ax.imshow(np.ma.masked_invalid(np.where(a.isna(),np.nan,(a<.05).astype(float))),vmin=0,vmax=1,cmap='Greys',aspect='auto');ax.set_yticks(range(9),[LABEL[m] for m in FM],fontsize=6);ax.set_xticks(range(len(co)),[NAMES.get(c,c) for c in co],rotation=40,ha='right',fontsize=6);ax.set_title(task,fontsize=8)
    for ax in list(axs.flat)[len(tasks):]:ax.set_axis_off()
    write(fig,'FigureS2_absolute_performance')
    ids=table(5,'S5_identification_scores');performance_grid(ids,'FigureS4_primary_identification',[(ids[ids.task.eq('top1')],'Primary frozen Re-ID'),(ids[ids.task.eq('I_diff')],'Within- versus between-participant similarity')])
    transfer=read('Figure3_source_transfer');loso=read('Figure3_LOSO_scores')
    fig,axs=plt.subplots(2,2,figsize=(12,10));fig.subplots_adjust(left=.14,right=.93,top=.94,bottom=.16,hspace=.7,wspace=.65)
    for j,task in enumerate(['MEM','EF']):
        q=transfer[transfer.task.eq(task+'_axis_to_'+task)].copy()
        q['setting']=q.source.map(NAMES)+' → '+q.cohort.map(NAMES)
        a=q.pivot(index='model',columns='setting',values='score').reindex(index=FM)
        ax=axs[0,j];im=ax.imshow(a,vmin=-.6,vmax=.6,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(len(a.columns)),a.columns,rotation=55,ha='right',fontsize=6);ax.set_yticks(range(9),[LABEL[m] for m in FM],fontsize=7);title(ax,'AB'[j],task+': source-fitted transfer')
        q=loso[loso.domain.eq(task)];a=q.pivot(index='model',columns='cohort',values='rho').reindex(index=METHODS)
        ax=axs[1,j];im=ax.imshow(a,vmin=-.6,vmax=.6,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(len(a.columns)),[NAMES.get(c,c) for c in a.columns],rotation=35,ha='right',fontsize=7);ax.set_yticks(range(len(METHODS)),[base.SHORT[m] for m in METHODS],fontsize=7);title(ax,'CD'[j],task+': held-out intervention cohorts')
    cax=fig.add_axes([.35,.04,.4,.012]);fig.colorbar(im,cax=cax,orientation='horizontal',label='Spearman ρ');write(fig,'FigureS5_complete_cross_cohort_transfer')
    arm=read('Figure4A_arm_scores');change=read('Figure4B_cognitive_change_input_differences');fu=read('Figure4C_baseline_imaging_followup_increment');delta=read('Figure4D_longitudinal_imaging_change_increment')
    fig,axs=plt.subplots(2,2,figsize=(11,9));fig.subplots_adjust(left=.15,right=.97,top=.94,bottom=.13,hspace=.6,wspace=.65)
    ax=axs[0,0]
    for ci,co in enumerate(RCT_COHORTS):
        for mi,model in enumerate(METHODS):
            for ii,mode in enumerate(['baseline','change','baseline_plus_change']):
                q=arm[arm.cohort.eq(co)&arm.model.eq(model)&arm['mode'].eq(mode)]
                if len(q):ax.scatter(q.auc,np.full(len(q),ci+(mi-5)*.04),s=12,color=C[model],marker=['o','s','^'][ii])
    ax.set_yticks(range(5),[NAMES[c] for c in RCT_COHORTS]);ax.set_xlabel('ARM: AUC');title(ax,'A','Baseline and longitudinal inputs');clean(ax)
    cloud(axs[0,1],change,'B  ΔCOG: input comparisons','Δρ (longitudinal − baseline)')
    cloud(axs[1,0],fu,'C  COG-FU: imaging increment','Δρ (clinical + imaging − clinical)')
    cloud(axs[1,1],delta,'D  ΔCOG: imaging increment','Δρ (clinical + imaging − clinical)')
    _original_legend(fig,[Line2D([],[],color='#444',marker=mk,ls='',label=label) for mk,label in [('o','Baseline'),('s','Follow-up − baseline'),('^','Baseline + change')]],3,gap_points=10);write(fig,'FigureS6_complete_intervention_questions')
    inputs=read('FigureS2_input_paired');clinical=read('FigureS2_clinical_increment')
    grid_cloud('FigureS7_complete_longitudinal_inputs',[(inputs[inputs.task.eq(task)],task,'Performance difference from baseline') for task in ['arm','MEM_change','EF_change']]+[(clinical,'Clinical increment','Performance difference from clinical reference')])
    direct=read('FigureS6_all_primary_source_transfers');focused=read('FigureS6_all_focused_readout_tests')
    grid_cloud('FigureS8_complete_posttraining_readout',[(direct[direct.source.eq(s)],NAMES[s]+' source transfer','Δρ (post-trained − frozen)') for s in ['adni','nacc']]+[(focused[focused.contrast.eq('adapted_vs_frozen')],'Same-readout comparison','Δρ (post-trained − frozen)'),(focused[focused.contrast.eq('readout_vs_native')],'Readout after adaptation','Δρ (connectivity − primary)')])
    align=read('FigureS3_all_alignment');inter=read('FigureS3_all_intervention_alignment')
    grid_cloud('FigureS9_embedding_alignment',[(align[align.setting.eq(s)&align.candidate.eq(c)],s+': '+c,'Δρ (aligned − unaligned)') for s in ['source_axis','local_followup'] for c in ['moments','coral']]+[(inter[inter.metric.eq(m)&inter.contrast.eq('coral_vs_source_pca_unaligned')],'Paired-visit '+m,'Difference from matched unaligned control') for m in ['rho','auc']])
    short=read('FigureS4_cognition_short_context');paired=read('FigureS4_all_paired_short_context')
    grid_cloud('FigureS10_input_context',[(short,'COG: 80-frame inputs','Δρ (80-frame − primary input)')]+[(paired[paired.task.eq(t)],t,'Performance difference (80-frame − primary)') for t in ['MEM_change','EF_change','arm']])
    phys=read('FigureS5_all_physiology')
    grid_cloud('FigureS11_physiology',[(phys[phys.cohort.eq(c)&phys.objective.eq('frozen')],NAMES[c]+': Frozen','Δρ (FM − reference)') for c in ['breathe','psopt','hrv']]+[(phys[phys.cohort.eq(c)&phys.reference.eq('frozen')],NAMES[c]+': Post-training','Δρ (post-trained − frozen)') for c in ['breathe','psopt','hrv']],3)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--supplementary',action='store_true',help='Also replot all supplementary result sets; aggregate scores/tests are unchanged.')
    args=parser.parse_args()
    if not (DATA/'Figure2_heatmap_scores_and_tests.csv').is_file():
        parser.error('Numerical inputs are not bundled. Set BRAINFM_RESULTS_DIR to your local result export (tables/ and figure_data/); see reproducibility/RESULTS.md.')
    OUT.mkdir(parents=True,exist_ok=True)
    plot_heatmap(read('Figure2_heatmap_scores_and_tests'));figure4();figure5()
    if args.supplementary:supplementary()
    (OUT/'provenance.json').write_text(json.dumps({'source_sha256':INPUTS,'statistical_tests_recomputed':False,'snapshot':'2026-11-05','supplementary_layout':'Numerical replot; see results/FIGURE_DATA.md for source mappings'},indent=2)+'\n')
    print(OUT)
if __name__=='__main__':main()
