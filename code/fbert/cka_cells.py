import sys, json, numpy as np
sys.path.insert(0,'/workspace/fbx'); import eval_samefolds as ev, eval_sixmod as sx
rest=sx.load_feats('/workspace/fbx/feat_cbd*.npz',{'fc_full':'fc_rs'})
task=sx.load_feats('/workspace/fbx/taskfeat_cbd*.npz',{f'fc_{m}':f'fc_{m}' for m in sx.MODS[1:]})
use=sorted(set(rest)&set(task)); ev.G_GRID=[0.85,0.97]
def cka(K,L):
    n=len(K); H=np.eye(n)-1/n; A=H@K@H; B=H@L@H; return float((A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B))
cells={k:[] for k in ['nofilt_g0.85','filt_g0.85','nofilt_g0.97','filt_g0.97']}
for m in sx.MODS:
    src=rest if m=='rs' else task
    fc=np.stack([src[s][f'fc_{m}'] for s in use]); res=[ev.per_subject(v) for v in fc]; Kf=ev.correlation_kernel(fc)
    for gi,g in enumerate([0.85,0.97]):
        Kp=ev.correlation_kernel(np.stack([r[1][gi] for r in res]))
        cells[f'nofilt_g{g}'].append(0.95*Kp+0.05*Kf); cells[f'filt_g{g}'].append(0.95*ev.spectral_filter(Kp)+0.05*Kf)
    print(m,flush=True)
out={}
for k,Ks in cells.items():
    v=[cka(Ks[i],Ks[j]) for i in range(6) for j in range(i+1,6)]; out[k]={'mean':float(np.mean(v)),'min':float(min(v)),'max':float(max(v))}
json.dump({'cells':out,'n':len(use),'note':'label-free; per-modality kernels incl. 5% full mix; filter = canonical tau 0.5 keep 150'},open('/workspace/fbx/CKA_CELLS_1117.json','w'),indent=1)
print(json.dumps(out,indent=1))
