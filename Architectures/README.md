# Architectures (Draw.io)

Open any `.drawio` file in [diagrams.net](https://app.diagrams.net) (File → Open) or the VS Code / Cursor Draw.io extension.

| File | Model | What it shows |
|------|--------|----------------|
| `01_Phase_MLP.drawio` | Phase-MLP baseline | UNetFiLM + continuous phase MLP FiLM + SVF; **4-ch** PatchGAN `[ref, DVF]` |
| `02_Warp_D.drawio` | Warp-D scenario | Same generator; **2-ch** PatchGAN `[warp(ref,DVF), target]` |
| `03_Dans_UNetCRB.drawio` | Dan’s GAN | UNetCRB (encoder CRBs, avg-pool, direct DVF); Warp-D PatchGAN |
| `04_FiLM_Detail_Phases_1_to_6.drawio` | FiLM only | Step-by-step Phase-MLP → concat → trunk → FiLM heads, worked example φ=1→6 |
| `05_Discriminator_PatchGAN.drawio` | PatchGAN D | Simple overview: 4-ch DVF-space vs 2-ch Warp-D + shared backbone |
| `06_Bottleneck.drawio` | UNetFiLM bot | What 8³ means; bot1→bot2→FiLM; ConvBlock guts; why two blocks |

Source of truth in code:
- Phase-MLP / Warp-D generator: `utilities/generator.py`, `utilities/film.py`, `utilities/svf.py`
- Discriminator modes: `utilities/discriminator.py`, `utilities/train_loop.py` (`d_input_mode`)
- Dan CRB: `Dan'sPaperGan/utilities/generator_crb.py`
