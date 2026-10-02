import sys,json,os,numpy as np
sys.path.insert(0,"/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT","/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import eval_dir_tcia3_iso2mm_v2 as v2
from dirlab_tre import landmarks_300
D="/media/abhishek/3CCA3CADCA3C6574/TCIA_4D-Lung/synth_g160_r3/pooled/train"
def feats(mu,lung):
    m=lung>0; z=np.where(m.any((1,2)))[0]
    return dict(vol_ml=m.sum()*8/1000, height_mm=(z.max()-z.min()+1)*2.0, dens=float(mu[m].mean()))
tr=[]
for s in range(1,83):
    p=f"{D}/S{s:02d}_06_to_01_pair.npy"
    if not os.path.exists(p): continue
    dvf=np.load(p,mmap_mode="r"); lung=np.load(f"{D}/S{s:02d}_Mask_Lung.npy"); mu=np.load(f"{D}/S{s:02d}_CT_06.npy",mmap_mode="r")
    mu=np.asarray(mu).squeeze(); m=lung>0; mag=np.linalg.norm(np.asarray(dvf)[m],axis=1)
    tr.append(dict(id=f"S{s:02d}",mean=float(mag.mean()),p95=float(np.percentile(mag,95)),**feats(mu,lung)))
te=[]
for c in range(1,11):
    pk=v2.pack_case(c); l0=v2.official_to_iso(landmarks_300(c,"T00"),pk); l5=v2.official_to_iso(landmarks_300(c,"T50"),pk)
    mag=np.linalg.norm((l5-l0)*2.0,axis=1)
    te.append(dict(id=f"C{c:02d}",mean=float(mag.mean()),p95=float(np.percentile(mag,95)),**feats(pk["mu"],pk["lung_iso"])))
json.dump(dict(train=tr,test=te),open(sys.argv[1],"w"),indent=1)
a=np.array([t["mean"] for t in tr])
print(f"TRAIN n={len(tr)} mean|motion| in lung: median {np.median(a):.2f}  p10 {np.percentile(a,10):.2f}  p90 {np.percentile(a,90):.2f}  max {a.max():.2f} mm")
print("TEST (landmarks):"); 
for t in te: print(f"  {t['id']} mean {t['mean']:.2f}  p95 {t['p95']:.2f}  vol {t['vol_ml']:.0f} ml  height {t['height_mm']:.0f}  dens {t['dens']:.4f}  rank-in-train {(a<t['mean']).mean()*100:.0f}%")
X=np.array([[t["vol_ml"],t["height_mm"],t["dens"]] for t in tr]); y=a
for i,n in enumerate(["vol_ml","height_mm","dens"]): print(f"corr(mean motion, {n}) = {np.corrcoef(X[:,i],y)[0,1]:+.2f}")
A=np.c_[X,np.ones(len(y))]; pred=[]
for i in range(len(y)):
    k=np.arange(len(y))!=i; w=np.linalg.lstsq(A[k],y[k],rcond=None)[0]; pred.append(A[i]@w)
pred=np.array(pred); print(f"leave-one-out R^2 (3 features -> motion) = {1-((y-pred)**2).sum()/((y-y.mean())**2).sum():.2f}")
