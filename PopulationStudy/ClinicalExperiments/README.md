# ClinicalExperiments

**Train fully on SPARE MC, test zero-shot on Clinical Varian** — same CRB stack as
[`InitialExperiments`](../InitialExperiments/) Experiments 1–2 (phase-only MSE, **no amplitude**).

| vs InitialExperiments | ClinicalExperiments |
|-----------------------|---------------------|
| Train | Leave-patient-out subsets | **All P1–P9** |
| Test | Other SPARE hold-outs | **Clinical Varian `CV_*`** |
| Amplitude | — | **Not used** (no shape norm, no `A_train`) |

Clinical data copy: [`../varian/`](../varian/) (`CV_P1` …). Varian SPARE layout notes:
`~/Documents/VoxelMap_Clinical/Varian.md`.

## Experiments

| Folder | Status | Summary |
|--------|--------|---------|
| [Experiment1](Experiment1/) | done | Full SPARE train · Varian/Elekta QC · **linear** phase · no amp |
| [Experiment2](Experiment2/) | retraining | Same · **cyclic** phase · (fix sys.path; was accidentally linear) |
| [Experiment3](Experiment3/) | done | **BothCRB** · cyclic · FOV/CBCT aug @ ¼ · Varian/Elekta QC |
| [Experiment4](Experiment4/) | done | **Inference-only** · score/correct DVF in **mm** (E3 ckpt) · small Elekta gain |
| [Experiment5](Experiment5/) | done | **Inference-only** · robust CT intensity norm (µ air/water) · large Elekta gain |
| [Experiment6](Experiment6/) | training | **128³** train · [Normal](Experiment6/Normal/) linear · [Cyclic](Experiment6/Cyclic/) later |

E1–E5 stay **zero-shot** (no clinical GT in training). E4/E5 do **not** fine-tune; they
change scoring units (E4) or test-time CT scaling (E5) on the E3 Both checkpoint.

### Experiment 3 — train augment modes (Both, full SPARE)

Per **train** sample, draw **exactly one** mode (equal weight):

| Mode | Prob | Transform |
|------|------|-----------|
| Normal | ¼ | as E1/E2 |
| Half-FOV | ¼ | random L or R cut (optional cut jitter) |
| CBCT noise | ¼ | full FOV + noise / streaks / cupping |
| **Half-FOV + noise** | ¼ | both (Elekta-like) |

Same FOV/noise on ref CT, target CT, and lung mask; MSE only in visible lung. **Val = Normal only.**
Zero-shot QC on Varian + Elekta after train.

### Experiment 4 — mm / scale rescoring (inference-only)

Uses **E3 Both** weights. Scores DVFs in packed voxels, millimetres, and
SPARE-scale-corrected voxels. **Result:** Elekta cos 0.20→0.22 only — not the main fix.
See [Experiment4/README.md](Experiment4/README.md).

### Experiment 5 — robust CT intensity norm (inference-only)

Uses **E3 Both** weights. Replaces min–max CT scaling with percentile / **µ air–water**
anchors at QC. **Result:** Elekta cos **0.20→0.51**, beat 59%→82%. Still zero-shot
(no clinical labels). See [Experiment5/README.md](Experiment5/README.md).

## Queue (all remaining ClinicalExperiments work)

```bash
# already launched; log:
#   ClinicalExperiments/logs/queue_remaining.log
#   ClinicalExperiments/logs/QUEUE_STATUS.md
bash ClinicalExperiments/scripts/queue_remaining.sh
```

Order: wait E2 enc→dec → train E2 Both → wait E1/E3 Both → QC backlog (E1 Both, E2 all archs, E3 Both) on Varian+Elekta.


## Shared pipeline (E1/E2 recipe)

- **EncoderCRB / DecoderCRB / BothCRB** — lung-masked MSE vs Elastix DVF
- Conditioning: `(t_ref, t_tgt)` only
- 100 epochs, lr `1e-4`, 64³ patches (16 train / 8 val per volume)
- Val = 10% **pair** split from train patients (for full SPARE: pairs from all 9)

## Clinical test prerequisites

Clinical Varian has `GTVol_01…10` + masks under `Evaluation/`, not Elastix pairs.
Before test QC:

1. Pad lung mask (same as SPARE — `scripts/viz_mask_padding.py`)
2. Elastix phase pairs on `GTVol_*` — `ClinicalExperiments/scripts/prepare_clinical_dvf_library.py`
3. Pack to **same 128³ grid** as SPARE train (`prepare_dvf_library.py` recipe)
4. GT-warp sanity (residual ≪ identity)

## Metrics

Same as InitialExperiments: L1, L1/zero, cosine vs Elastix. Oracle-α may be reported as a
**diagnostic** only — not a model input.

## Related

- In-cohort baselines: [`../InitialExperiments/`](../InitialExperiments/), [`../results.md`](../results.md)
- Amplitude work (out of scope here): [`../CRBExperiments/`](../CRBExperiments/)
