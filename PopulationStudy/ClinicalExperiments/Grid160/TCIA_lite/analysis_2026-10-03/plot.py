import re,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B="/home/abhishek/Voxel_GAN/PopulationStudy/ClinicalExperiments/Grid160/"
runs={"TCIA3.5_hybrid (GPU0, no aug)":B+"TCIA3.5_hybrid/logs/train.log","A2_full160_aug_hybrid_s2 (GPU1)":B+"TCIA_lite/run_A2_full160_aug_hybrid_s2/logs/train.log"}
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for a,(n,p) in zip(ax,runs.items()):
    e,t,v=[],[],[]
    for l in open(p):
        m=re.search(r"Epoch: (\d+) \| train MAE: ([\d.]+) \| val MAE: ([\d.]+)",l)
        if m: e.append(int(m[1]));t.append(float(m[2]));v.append(float(m[3]))
    a.plot(e,t,"o-",label="train MAE");a.plot(e,v,"s-",label="val MAE")
    i=v.index(min(v));a.annotate(f"best val {v[i]:.3f}",(e[i],v[i]),textcoords="offset points",xytext=(0,10),ha="center")
    a.set_title(n);a.set_xlabel("epoch");a.set_ylabel("MAE");a.grid(alpha=.3);a.legend()
    print(n,list(zip(e,t,v)))
plt.tight_layout();plt.savefig("/tmp/claude-1003/-home-abhishek-Voxel-GAN/05a92188-ab48-4025-bc01-8263a8e75ec9/scratchpad/training_curves.png",dpi=120)
