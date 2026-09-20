"""Summarize matched-input Stage A validation and render deterministic examples."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.metrics import compute_metrics
root=Path(sys.argv[1])
keys=['union02','union0','las']
reports={k:json.loads((root/(k+'.json')).read_text()) for k in keys}
arrays={}
for k in keys:
    with np.load(root/(k+'.npz')) as archive:
        arrays[k]={n:archive[n] for n in archive.files}
assert len({r['input_sha256'] for r in reports.values()})==1
base=arrays['union0']; rr=reports['union0']
for k in keys:
    for n in ['targets','affordance_ids','category_ids','points','visual_indices','visual_images']:
        assert np.array_equal(base[n],arrays[k][n]),(k,n)
    assert [r['object_id'] for r in reports[k]['per_object']]==[r['object_id'] for r in rr['per_object']]
summary={'input_sha256':rr['input_sha256'],'models':{},'per_affordance':{},'per_category':{},'examples':[]}
for k in keys:
    rep=reports[k]; a=arrays[k]
    row={n:rep[n] for n in ['metrics','checkpoint_epoch','low_differentiation_objects','distinct_gt_objects','prediction_pair_mae','gt_pair_mae','near_identical_objects','cue_prediction_mae','rng_prediction_mae']}
    maes=[]
    for record in rep['per_object']:
        sl=slice(record['start'],record['end'])
        maes.append(float(np.abs(a['original'][sl]-a['targets'][sl]).mean()))
    row['object_mae']=maes
    row['macro_affordance_aiou']=float(np.mean([v['aiou'] for v in rep['per_affordance'].values()]))
    if 'selector' in rep:
        row['selector']=rep['selector']
        maps=a['basis_maps']; pairs=[np.abs(maps[:,:,i]-maps[:,:,j]).mean(1) for i in range(8) for j in range(i+1,8)]
        row['basis_pair_mae']=float(np.mean(pairs))
    summary['models'][k]=row
for group in ['affordance','category']:
    for label in rr['per_'+group]:
        summary['per_'+group][label]={k:reports[k]['per_'+group][label] for k in keys}
d=np.array(summary['models']['union0']['object_mae'])-np.array(summary['models']['union02']['object_mae'])
summary['object_mae_comparison']={'improved':int((d<-1e-6).sum()),'worsened':int((d>1e-6).sum()),'tied':int((np.abs(d)<=1e-6).sum()),'mean_delta':float(d.mean())}
plt.rcParams.update({'font.size':9})
fig,axes=plt.subplots(1,3,figsize=(13,4))
labels=['Union 0.2','Union 0','LAS']
for idx,k in enumerate(keys):
    rec=reports[k]['per_object']; x=np.array([t['gt_pair_mae'] for t in rec]); y=np.array([t['prediction_pair_mae'] for t in rec])
    axes[idx].scatter(x,y,s=8,alpha=.35)
    axes[idx].plot([0,.8],[0,.8],'k--',lw=1,label='GT difference')
    axes[idx].plot([0,.8],[0,.2],color='red',lw=1,label='25% of GT')
    axes[idx].set(xlim=(0,.8),ylim=(0,.8),xlabel='GT pair MAE',ylabel='Prediction pair MAE',title=labels[idx])
axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'object_differentiation.png',dpi=150);plt.close(fig)
def cloud(ax,xyz,value,title):
    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=value,cmap='viridis',vmin=0,vmax=1,s=3,rasterized=True)
    ax.view_init(elev=22,azim=45); ax.set_box_aspect((1,1,1));ax.set_axis_off(); ax.set_title(title,fontsize=8)
for category in ['Mug','Table','TrashCan','Bottle']:
    cid=rr['category_vocabulary'].index(category)
    oi=next(i for i,rec in enumerate(rr['per_object']) if rec['category_id']==cid)
    rec=rr['per_object'][oi]; start,end=rec['start'],rec['end']; n=end-start
    xyz=base['points'][oi]
    fig=plt.figure(figsize=(16,3*n+6))
    grid=fig.add_gridspec(n+2,8)
    for j,p in enumerate(range(start,end)):
        aid=int(base['affordance_ids'][p]); name=rr['affordance_vocabulary'][aid]
        imageidx=np.flatnonzero(base['visual_indices']==p)[0]
        image=base['visual_images'][imageidx].transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])
        ax=fig.add_subplot(grid[j,0]);ax.imshow(np.clip(image,0,1));ax.axis('off');ax.set_title(name)
        cloud(fig.add_subplot(grid[j,1:3],projection='3d'),xyz,base['targets'][p],'GT')
        cloud(fig.add_subplot(grid[j,3:5],projection='3d'),xyz,arrays['union02']['original'][p],'Union 0.2')
        cloud(fig.add_subplot(grid[j,5:7],projection='3d'),xyz,base['original'][p],'Union 0')
        cloud(fig.add_subplot(grid[j,7],projection='3d'),xyz,arrays['las']['original'][p],'LAS')
    for row,k in enumerate(['union02','union0']):
        for b in range(8):
            cloud(fig.add_subplot(grid[n+row,b],projection='3d'),xyz,arrays[k]['basis_maps'][oi,:,b],f'{k} basis {b}')
    fig.suptitle(category+' | first validation object in category | '+rec['object_id'][:24]+'\nAll point colors use [0,1]. Basis indices are not aligned across models.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95)); filename=category.lower()+'_predictions_bases.png';fig.savefig(root/filename,dpi=120);plt.close(fig)
    example={'category':category,'object_id':rec['object_id'],'file':filename,'affordances':[rr['affordance_vocabulary'][int(base['affordance_ids'][p])] for p in range(start,end)],'alpha':{k:arrays[k]['alpha'][start:end].tolist() for k in ['union02','union0']},'metrics':{k:compute_metrics(arrays[k]['original'][start:end],base['targets'][start:end]) for k in keys}}
    summary['examples'].append(example)
(root/'summary.json').write_text(json.dumps(summary,indent=2,default=float))
print(json.dumps({**summary,'models':{k:{n:v for n,v in row.items() if n!='object_mae'} for k,row in summary['models'].items()}},default=float))
