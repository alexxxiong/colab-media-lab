"""Offline analysis and figures from the included, sanitized measurements."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT/'data'/name).open(encoding='utf-8',newline='') as stream: return list(csv.DictReader(stream))


def compute():
    observations={r['sample_id']:r for r in rows('observations.csv')}
    startup=rows('startup-observation.csv')
    total=float(startup[-1]['seconds'])
    first=float(observations['M43']['request_seconds'])
    rate=10.05/3600
    ledger=rows('research-session-costs.csv')
    summary={
        'cpu_only_analysis':True,'formal_benchmark_completed':False,
        'startup_observation':{'n':1,'time_to_first_validated_media_seconds':total,
            'pre_request_preparation_seconds':total-first,'first_request_with_validation_seconds':first,
            'time_to_first_media_cu_estimate':total*rate,'preparation_cu_estimate':(total-first)*rate,
            'first_request_cu_estimate':first*rate},
        'duration_cu_estimate':sum(float(r['duration_cu_estimate']) for r in ledger),
        'balance_delta_cu_observed':sum(float(r['balance_delta_cu_observed']) for r in ledger),
        'accepted_main_clips':0,'controlled_hot_sample_count':0,
        'warning':'Research observations differ in prompt, seed and cache state. No reliable P95 or production yield inferred.'}
    summary['unreconciled_difference_cu']=summary['duration_cu_estimate']-summary['balance_delta_cu_observed']
    return summary,observations,startup


def figures(observations,startup):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'colab-media-lab'})
    from matplotlib import pyplot as plt
    folder=ROOT/'figures'
    folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(folder/(name+'.svg'),metadata={'Date':None})
        fig.savefig(folder/(name+'.png'),dpi=160)
        plt.close(fig)
    samples=[observations[n] for n in ('P04','M42','M43','M44')]
    labels=[r['sample_id']+' | '+r['width']+'x'+r['height'] for r in samples]
    times=[float(r['generation_seconds']) for r in samples]
    fig,ax=plt.subplots(figsize=(9,4.2))
    ax.barh(labels,times,color=['#5487ce','#7585a8','#b17b71','#6c9f98'])
    for n,value in enumerate(times): ax.text(value+2,n,f'{value:.2f}s',va='center')
    ax.set_xlim(0,max(times)*1.2)
    ax.set_xlabel('Generation time (seconds), research samples; cache state differs')
    ax.set_title('Technical media results; all main candidates failed content acceptance')
    ax.grid(axis='x',alpha=.2)
    ax.set_axisbelow(True)
    save(fig,'generation-times')
    values=[float(r['seconds']) for r in startup[:-1]]
    values.append(float(startup[-1]['seconds'])-sum(values))
    names=['Kernel ready','Drive auth/mount','Environment restore','Weight staging/checks',
           'Framework init','First request + validation','Other import/staging gaps']
    fig,(ax,legend_ax)=plt.subplots(2,1,figsize=(10,4.2),gridspec_kw={'height_ratios':[2,1]})
    left=0
    for name,value,color in zip(names,values,['#8ea8d4','#b3bfda','#85b4ad','#579a92','#baac99','#b77f75','#cad1da']):
        ax.barh(['R02'],[value],left=left,label=f'{name}: {value:.1f}s',color=color)
        left+=value
    ax.set_xlabel('Seconds from observed assignment to first validated media')
    ax.set_title('One research startup observation, not the planned three-run cold benchmark')
    handles,labels=ax.get_legend_handles_labels()
    legend_ax.axis('off')
    legend_ax.legend(handles,labels,loc='center',ncol=2,fontsize=9,frameon=False)
    save(fig,'startup-observation')
    telemetry=rows('resources/R02-startup-resources.csv')
    minutes=[float(r['elapsed_seconds'])/60 for r in telemetry]
    fig,axes=plt.subplots(3,1,figsize=(9,6.5),sharex=True)
    for ax,key,label,scale,color in zip(axes,
            ['gpu_util_percent','gpu_used_mib','power_watts'],['GPU %','Board VRAM (GiB)','Board power (W)'],
            [1,1024,1],['#5487ce','#6c9f98','#b17b71']):
        ax.plot(minutes,[float(r[key])/scale if r[key] else float('nan') for r in telemetry],lw=.8,color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_title('R02 whole-board 1 Hz samples; includes research and prompt-preparation waits')
    axes[-1].set_xlabel('Elapsed minutes from first startup telemetry sample')
    save(fig,'r02-resources')


def main():
    summary,observations,startup=compute()
    (ROOT/'data/analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    try:
        figures(observations,startup)
        summary['figures_generated']=True
    except ImportError:
        summary['figures_generated']=False
        summary['figure_note']='Install matplotlib to regenerate figures; committed figures remain available.'
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
