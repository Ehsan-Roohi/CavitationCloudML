"""Report all registered outcomes, including adverse transfer and strong baselines."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cavitation_stress_20260912 import B,O,CASES,corrupt

def aggregate(rows):
    result=[]
    for group in ['training','development']:
        conditions=sorted(set((r['kind'],r['level']) for r in rows))
        for kind,level in conditions:
            selected=[r for r in rows if r['group']==group and (r['kind'],r['level'])==(kind,level)]
            methods=sorted(set(m for r in selected for m in r['metrics']))
            for method in methods:
                for cls in ['attached_2d','disconnected_2d']:
                    bycase={}
                    for r in selected:
                        if method in r['metrics']:
                            value=r['metrics'][method][cls]['dice']
                            if value is not None:bycase.setdefault(r['case'],[]).append(value)
                    vals=[np.mean(v) for v in bycase.values()]
                    result.append(dict(group=group,kind=kind,level=level,method=method,cls=cls,mean_case_dice=float(np.mean(vals)),cases=len(vals),case_values={k:float(np.mean(v)) for k,v in bycase.items()}))
    return result

def main():
    original=json.loads((O/'RESULTS.json').read_text());strong=json.loads((O/'strong_controls/RESULTS.json').read_text())
    summary=aggregate(original['rows']+strong['rows'])
    (O/'SUMMARY.json').write_text(json.dumps({'aggregation':'mean across cases of mean seed Dice; no frame-level significance claim','rows':summary},indent=2))
    with (O/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    dev=[r for r in summary if r['group']=='development']
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    colors={'neural':'#0072b2','raster4':'#d55e00','raster8':'#999999','raster4_smooth':'#009e73','raster8_smooth':'#cc79a7','neural_filled':'#0072b2','raster4_filled':'#009e73','raster8_filled':'#cc79a7'}
    for col,kind in enumerate(['noise','missing','coarse']):
        for row,cls in enumerate(['attached_2d','disconnected_2d']):
            ax=axs[row,col]
            for method in sorted(set(r['method'] for r in dev if r['kind']==kind)):
                data=sorted([r for r in dev if r['kind']==kind and r['cls']==cls and r['method']==method],key=lambda r:r['level'])
                ax.plot([r['level'] for r in data],[r['mean_case_dice'] for r in data],marker='o',label=method.replace('_',' '),color=colors[method],ls='--' if 'filled' in method else '-')
            ax.set_ylim(0,1.03);ax.grid(alpha=.2);ax.set_xlabel({'noise':'Noise standard deviation in alpha','missing':'Deleted sample fraction','coarse':'Block-averaging factor'}[kind]);ax.set_ylabel(('Attached' if row==0 else 'Detached cloud')+' Dice');ax.legend(fontsize=7)
    fig.suptitle('Frozen model stress tests | Cases 19, 24, 23 | Clean native weak references\nCase-macro means; stochastic conditions use three fixed seeds',fontsize=12)
    fig.savefig(O/'ROBUSTNESS.png',dpi=170);plt.close(fig)
    # Prospective worst-error frame for each new case, without changing predictions.
    P=B/'prospective_archive_v1';fig,axs=plt.subplots(2,3,figsize=(13,5.3),layout='constrained');selections=[]
    for row,name in enumerate(['Case_8','Case_20']):
        with np.load(P/'data_audit'/f'{name}_raster.npz') as z:a=z['alpha_v'];w=z['wall'];x=z['x'];y=z['y']
        with np.load(P/f'{name}_prediction.npz') as z:t=z['target'];p=z['neural'];b=z['raster4'];times=z['times']
        i=int(np.argmax(((p!=t)&(t!=255)).sum((1,2))));selections.append(dict(case=name,frame=i,time=float(times[i]),selection='largest number of misclassified valid pixels'))
        for col,label in enumerate([t,p,b]):
            ax=axs[row,col];ax.imshow(np.ma.masked_where(w,a[i]),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='Blues',vmin=0,vmax=1)
            for c,color in [(1,'#e69f00'),(2,'#cc3299')]:
                mask=(label[i]==c)&~w
                if mask.any():ax.contour(x,y,mask,levels=[.5],colors=color,linewidths=.9)
            ax.contourf(x,y,w,levels=[.5,1.5],colors='white');ax.contour(x,y,w,levels=[.5],colors='black',linewidths=.6)
            ax.set_title(f'{name.replace("_"," ")} t={times[i]:g}s | '+['Native reference','Frozen U-Net','Raster connectivity'][col],fontsize=9);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Previously unused cases | Worst-error frames retained\nOrange: attached; magenta: detached | No retraining',fontsize=12)
    fig.savefig(O/'PROSPECTIVE_FAILURES.png',dpi=180);plt.close(fig)
    (O/'FIGURE_SELECTION.json').write_text(json.dumps(selections,indent=2))
    lines=['# Frozen cavitation robustness and archive-transfer evaluation','','All metrics compare to clean native alpha>=0.20 weak labels, not expert truth. Cases 19/24/23 were exposed development cases. Training cases are excluded from the table below. Seed averaging precedes case averaging.','','| Condition | Method | Attached Dice | Cloud Dice |','|---|---|---:|---:|']
    for kind,level,method in sorted(set((r['kind'],r['level'],r['method']) for r in dev)):
        pair={r['cls']:r['mean_case_dice'] for r in dev if (r['kind'],r['level'],r['method'])==(kind,level,method)}
        lines.append(f'| {kind} {level:g} | {method} | {pair["attached_2d"]:.3f} | {pair["disconnected_2d"]:.3f} |')
    lines+=['','## Interpretation','Clean fields do not establish an advantage over direct connectivity. Neural resistance to unfiltered noise must be compared with the smoothed controls, not only an unfiltered threshold. Missing-data results include identical nearest-observed imputation for both methods. These are synthetic measurement corruptions, not a new CFD regime or experimental validation.','', '## Previously unused archive cases']
    prospective=json.loads((P/'RESULTS.json').read_text())
    for r in prospective['cases']:
        q=r['metrics']['neural'];lines.append(f'- {r["case"]}: attached/cloud Dice {q["attached_2d"]["dice"]:.3f}/{q["disconnected_2d"]["dice"]:.3f}; {r["cloud_reference_pixels"]} cloud reference pixels. Raster4 Dice is 1.0/1.0 in both cases.')
    lines+=['','The model was frozen before fields for cases 8 and 20 were opened; case IDs were registered in advance. These are unseen-case transfer checks within the same archive, not independently annotated truth. Case 20 exposes poor cloud transfer. Do not replace this result with a selected favorable frame.','', 'No model retraining or threshold tuning was performed. Strong control Gaussian sigma=0.7 pixels was fixed before its evaluation, not optimized per case. Geometry is kept exact in all corruptions. Raw reference connectivity and raster inference baseline differ explicitly.']
    (O/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
if __name__=='__main__':main()
