# Architectures (Draw.io)

Open any `.drawio` file in [diagrams.net](https://app.diagrams.net) (File → Open) or the VS Code / Cursor Draw.io extension.

| File | Model | What it shows |
|------|--------|----------------|
| `01_Phase_MLP.drawio` | Phase-MLP baseline | UNetFiLM + continuous phase MLP FiLM + SVF; **4-ch** PatchGAN `[ref, DVF]` |
| `02_Warp_D.drawio` | Warp-D scenario | Same generator; **2-ch** PatchGAN `[warp(ref,DVF), target]` |
| `03_Dans_UNetCRB.drawio` | Dan’s GAN | UNetCRB (encoder CRBs, avg-pool, direct DVF); Warp-D PatchGAN |

Source of truth in code:
- Phase-MLP / Warp-D generator: `utilities/generator.py`, `utilities/film.py`, `utilities/svf.py`
- Discriminator modes: `utilities/discriminator.py`, `utilities/train_loop.py` (`d_input_mode`)
- Dan CRB: `Dan'sPaperGan/utilities/generator_crb.py`
