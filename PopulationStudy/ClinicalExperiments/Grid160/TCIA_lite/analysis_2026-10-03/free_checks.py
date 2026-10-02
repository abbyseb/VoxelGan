import sys,os,json,numpy as np,torch
sys.path.insert(0,"/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/scripts")
os.environ.setdefault("DIRLAB_ROOT","/home/abhishek/Voxel_GAN/DIR EXPERIMENTS/data/dirlab_packs")
import mirror_tta_iso2mm as mt, eval_dir_tcia3_iso2mm_v2 as v2
from rescore_oracle_iso2mm_v3 import load_ckpt
S=sys.argv[1]; G="/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160"; L=G+"/TCIA_lite"
ck=lambda run,e:f"{L}/{run}/DecoderCRB/checkpoints/epoch_{e:03d}.pt"
def avg(paths,out):
    raws=[torch.load(p,map_location="cpu",weights_only=False) for p in paths]; r=raws[0]
    sd={k:(sum(x["generator"][k].float() for x in raws)/len(raws)).to(r["generator"][k].dtype) for k in r["generator"]}
    torch.save({**r,"generator":sd},out); return out
M={"s1@40":ck("run_A2_full160_aug_hybrid",40),"s2@40":ck("run_A2_full160_aug_hybrid_s2",40),
   "s1_avg36-40":avg([ck("run_A2_full160_aug_hybrid",e) for e in range(36,41)],S+"/s1_avg.pt"),
   "s2_avg36-40":avg([ck("run_A2_full160_aug_hybrid_s2",e) for e in range(36,41)],S+"/s2_avg.pt"),
   "T35hyb@46":G+"/TCIA3.5_hybrid/DecoderCRB/checkpoints/epoch_046.pt"}
dev=torch.device("cuda:0"); cases=range(1,11); packs={c:v2.pack_case(c) for c in cases}; F={}
for n,p in M.items():
    g=load_ckpt(__import__("pathlib").Path(p),dev); F[n]={"plain":{},"mirror":{}}
    for c in cases: F[n]["plain"][c],F[n]["mirror"][c]=mt.predict_mirror(g,dev,packs[c]["mu"])
    del g; torch.cuda.empty_cache()
for name,mem in {"ENS s1+s2":["s1_avg36-40","s2_avg36-40"],"ENS s1+s2+T35hyb":["s1_avg36-40","s2_avg36-40","T35hyb@46"]}.items():
    F[name]={v:{c:np.mean([F[m][v][c] for m in mem],0) for c in cases} for v in ("plain","mirror")}
rows={}
for n in F:
    for v in ("plain","mirror"):
        per={c:mt.tre(F[n][v][c],c,packs[c])[1] for c in cases}; rows[f"{n}|{v}"]=per
        print(f"{n:22s} {v:6s} TRE300 {np.mean(list(per.values())):.3f}  C08 {per[8]:.2f}",flush=True)
json.dump({k:{str(c):x for c,x in v.items()} for k,v in rows.items()},open(S+"/free_checks.json","w"),indent=1)
