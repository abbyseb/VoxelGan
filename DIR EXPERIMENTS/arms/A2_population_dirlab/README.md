# A2 — Population VoxelMap (DIR-Lab)

VoxelMap trained on the **DIR-Lab population** (leave-one-out / train-on-other-cases),  
then evaluated TRE on the held-out case. Real DIR 4D-CT → DRR → VoxelMap — no synthesizer.

Role: population prior on the **same domain** as the test set; sits between patient oracle (A1) and synth (A3).

Status: scaffold — LOO protocol + train + TRE TBD.
