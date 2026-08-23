# Anatomy pretrain data (LIDC-IDRI)

Chest CT from [TCIA LIDC-IDRI](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX) for a standalone anatomy autoencoder. Not SPARE 4D-CT; no DVF labels.

| | |
|---|---|
| License | [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) |
| TCIA usage | https://wiki.cancerimagingarchive.net/x/c4hF |
| Full collection | ~1010 patients, 1018 CT series, ~128 GB |
| Default subset | 40 patients, one series each |

DICOM is **not** committed (`data/` in `.gitignore`).

```bash
python PopulationStudy/AnatomyPretrain/download_lidc_subset.py          # 40 patients
python PopulationStudy/AnatomyPretrain/download_lidc_subset.py --n-patients 100
python PopulationStudy/AnatomyPretrain/download_lidc_subset.py --all    # ~128 GB
```

Cite Armato et al. (TCIA LIDC-IDRI) if this data is used in a paper.

## Lung masks (R231)

[JoHof/lungmask](https://github.com/JoHof/lungmask) U-net(R231), left+right collapsed to binary. Hofmanninger et al. 2020.

```bash
# DICOM → CT.mha + Mask_Lung.mha
python PopulationStudy/AnatomyPretrain/prepare_lidc_masks.py

# same PyQtGraph padding viewer as SPARE
python PopulationStudy/scripts/viz_mask_padding.py --lidc --gui
```

Packed 128³ volumes for Experiment 4 live under `PopulationStudy/Experiment4/data/lidc128/` (see that README).

