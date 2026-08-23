# Initial Experiments (E1–E6)

Leave-patient-out CRB MSE runs on SPARE. E1–E5 used the old 128³ anisotropic crops; **E6** repeats the E1-split Decoder baseline on the P0-A common isotropic grid.

| Folder | Split | Idea |
|--------|-------|------|
| [Experiment1](Experiment1/) | Train P1,P2,P6–P9 · hold-out P3–P5 | Joint MSE CRB (aniso 128³) |
| [Experiment2](Experiment2/) | Train P1,P3–P5 · hold-out P7,P9 | Same, new split |
| [Experiment3](Experiment3/) | Same as E2 | Episodic `L_s + 2.5 L_q` |
| [Experiment4](Experiment4/) | Same as E2 | Frozen LIDC anatomy encoder |
| [Experiment5](Experiment5/) | Same as E1 | LIDC anatomy on E1 split |
| [Experiment6](Experiment6/) | Same as E1 | **A0 baseline on `data_iso` 2 mm 160³** |

Write-ups: [`../results.md`](../results.md) · next steps: [`../ChangesNeeded.md`](../ChangesNeeded.md) · amp companion: [`../CRBExperiments/Experiment1/`](../CRBExperiments/Experiment1/)
