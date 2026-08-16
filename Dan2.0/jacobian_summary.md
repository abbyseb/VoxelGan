# Dan 2.0 — negative Jacobian %

`negJ%` = fraction of voxels with `det(I+∇u) < 0` (finite-diff). Lower is better.

| Model | Dir full % | Dir lung % | Leave-out lung % | Max lung % |
|-------|------------|------------|------------------|------------|
| EncoderCRB/5_9 | 0.001 | 0.000 | 0.000 | 0.000 |
| EncoderCRB/3_6 | 0.000 | 0.000 | 0.000 | 0.000 |
| EncoderCRB/3_6_8 | 0.006 | 0.000 | 0.000 | 0.000 |
| DecoderCRB/5_9 | 0.002 | 0.000 | 0.000 | 0.000 |
| DecoderCRB/3_6 | 0.001 | 0.000 | 0.000 | 0.000 |
| DecoderCRB/3_6_8 | 0.002 | 0.000 | 0.000 | 0.000 |
| BothCRB/5_9 | 0.001 | 0.000 | 0.000 | 0.000 |
| BothCRB/3_6 | 0.000 | 0.000 | 0.000 | 0.000 |
| BothCRB/3_6_8 | 0.000 | 0.000 | 0.000 | 0.000 |
| DecoderFiLM/5_9 | 0.000 | 0.000 | 0.000 | 0.000 |
| DecoderFiLM/3_6 | 0.000 | 0.000 | 0.000 | 0.000 |
| DecoderFiLM/3_6_8 | 0.000 | 0.000 | 0.000 | 0.000 |
| DecoderCRBBot/5_9 | 0.024 | 0.000 | 0.000 | 0.000 |
| DecoderCRBBot/3_6 | 0.002 | 0.000 | 0.000 | 0.000 |
