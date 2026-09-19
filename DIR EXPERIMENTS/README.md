# DIR EXPERIMENTS — VoxelMap TRE on DIR-Lab

Primary KPI: **TRE in mm** on DIR-Lab landmarks (not Elastix L1/cos).

## Landmark sets (KPI)

Report **both** on T00→T50 (unless noted):

| Set | Role |
|-----|------|
| **75-pt** Sampled4D | Primary internal KPI / trajectory subsample |
| **300-pt** extreme | Paper-comparable DIR-Lab headline set |

Do not mix 75 and 300 when comparing to literature. `--check` still uses 300-pt identity vs published for pack QA.

Data: `data/dirlab_packs/Case{N}Pack` (official packs).  
Processed SPARE-style volumes also live under `PopulationStudy/DIR-Experiments/data/P*_DIR` (CRB packing / prior work).

TRE harness: [`scripts/dirlab_tre.py`](scripts/dirlab_tre.py) · A1 eval: [`scripts/eval_a1_tre.py`](scripts/eval_a1_tre.py)  
```bash
export DIRLAB_ROOT="$(pwd)/data/dirlab_packs"
python scripts/dirlab_tre.py --check
python scripts/eval_a1_tre.py --case 1 --gpu 0   # prints 75 + 300
```

Field convention for `--dvf`: `(nz, ny, nx, 3)` as `(dx, dy, dz)` **voxels**,  
`pred = src + dvf(src)`. If your warp is ITK/`grid_sample` style (`out(p)=moving(p+u)`),  
invert before calling (see script docstring) — wrong sign looks like mediocre TRE, not a crash.

---

## Testing arms (VoxelMap)

| ID | Arm | What it is | Role |
|----|-----|------------|------|
| **A0** | [`arms/A0_identity/`](arms/A0_identity/) | Static phase, **no motion** (identity DVF) | Lower bound |
| **A1** | [`arms/A1_oracle_dirlab/`](arms/A1_oracle_dirlab/) | VoxelMap trained on the **patient’s own** DIR-Lab 4D-CT | Best possible, clinically unrealistic |
| **A2** | [`arms/A2_generic_spare/`](arms/A2_generic_spare/) | VoxelMap trained on **SPARE population** motion (no patient-specific synth) | Does conditioning add anything? |
| **A3** | [`arms/A3_synth_conditioned/`](arms/A3_synth_conditioned/) | VoxelMap trained on **conditioned synthesizer** 4D-CT (e.g. G160-A1 Dec + µ from on-table CT) | **Main arm** |

### Optional

| ID | Arm | What it is |
|----|-----|------------|
| **A4** | [`arms/A4_mismatched_conditioning/`](arms/A4_mismatched_conditioning/) | Synth conditioned on a **different** patient’s static CT — specificity control |
| **A5** | [`arms/A5_synth_cbct_finetune/`](arms/A5_synth_cbct_finetune/) | A3 + fine-tune on CBCTs simulated from synthetic 4D-CTs — domain adaptation |

---

## Suggested order

1. **Harness** — `dirlab_tre.py --check` (pack QA).  
2. **A0** — report **75-pt** identity TRE on T00→T50.  
3. **A1 oracle** — stage DIR 4D-CT → VoxelMap train per case → **75-pt** TRE.  
4. **A2 generic** — SPARE-trained VoxelMap on DIR → **75-pt** TRE.  
5. **A3 synth** — G160-A1 (or chosen synthesizer) from mid-phase CT → synth 4D → VoxelMap → **75-pt** TRE.  
6. **A4 / A5** if A3 is competitive.

Leave-one-out vs train-on-all-but-test-case for A1/A2 should be fixed before claiming cohort numbers.

---

## Relation to `PopulationStudy/DIR-Experiments`

That folder is the **CRB DVF** transfer line (pack → E6 zero-shot L1/cos + early TRE).  
**This** folder is the **VoxelMap localization / TRE** line with the arms above. Reuse packs and landmark lessons; do not mix KPI tables without labeling the model.
