# Initial Experiments (E1–E5)

Leave-patient-out CRB MSE runs on SPARE before the amplitude / regrid work in `ChangesNeeded.md`.

| Folder | Split | Idea |
|--------|-------|------|
| [Experiment1](Experiment1/) | Train P1,P2,P6–P9 · hold-out P3–P5 | Joint MSE CRB |
| [Experiment2](Experiment2/) | Train P1,P3–P5 · hold-out P7,P9 | Same, new split |
| [Experiment3](Experiment3/) | Same as E2 | Episodic `L_s + 2.5 L_q` |
| [Experiment4](Experiment4/) | Same as E2 | Frozen LIDC anatomy encoder |
| [Experiment5](Experiment5/) | Same as E1 | LIDC anatomy on E1 split |

Write-ups: [`../results.md`](../results.md) · next steps: [`../ChangesNeeded.md`](../ChangesNeeded.md)
