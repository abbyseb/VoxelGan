# A3_synth_conditioned

Fresh arm folder for **R3-centred** DIR DRRs (`Geometry_SPARE` / A1 `Geometry.xml` OffsetY −2).

**Recipe (2026-09-16):** mid-phase R3 CT_06 → G160-A1 Decoder synth 10 phases → keep-HU downsample → **`DVF_sub = −u`** (Elastix fixed→moving; G160 warp uses pull `I(x+u)`) → SPARE DRR → NoFiLM 50 ep → TRE on **A1** ModelTraining (`eval_a3_tre.py --r3`).

```bash
cd "DIR EXPERIMENTS"
bash arms/A3_synth_conditioned/logs/pipeline_a3_r3_all_gpu1.sh
# log: arms/A3_synth_conditioned/logs/pipeline_a3_r3_all_gpu1.log
```

Previous (incorrect-orbit) runs: [`../Incorrect DRR/A3_synth_conditioned/`](../Incorrect%20DRR/A3_synth_conditioned/)
