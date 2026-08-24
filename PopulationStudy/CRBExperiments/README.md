# CRBExperiments

Amplitude-conditioned CRB ablations after Run 0 (H1 confirmed) and [`../AmplitudeConditioning.md`](../AmplitudeConditioning.md).

All on **`data_iso`** (2 mm, 160³), E1 split: train P1,P2,P6–P9 · hold-out P3–P5.

| Experiment | Arm | What |
|------------|-----|------|
| **[Experiment1](Experiment1/)** | A2 | Oracle `log(r_p)` + shape targets |
| **[Experiment2](Experiment2/)** | A1 | Shape targets only, `log(r̂)=0`, infer × `Ā_train` |
| **[Experiment3](Experiment3/)** | cyclic | Cyclic `(cos,sin)` phases + oracle amp |
| **[Experiment4](Experiment4/)** | δ_tgt | Cyclic + inhale/exhale descriptor + oracle amp |
| later | A3 / … | measured ΔV surrogate, `f` |

Baseline A0 (raw DVF, no amp): [`../InitialExperiments/Experiment6/`](../InitialExperiments/Experiment6/).
