LEARN-GUI DVF Visualisation
============================

Run folder: /home/abhishek/Voxel_GAN/VoxelMap_Experiments/runs/CV_P3_V_01/synth_g160_a1_dec

1. Open LEARN-GUI-Python and load this folder as the active run, OR
   Menu → DVF Visualisation and set:
   Patient folder: ModelTraining/test/CV_P3_V_01
   Model weights:  Models/weights_concatenated_nofilm_synth_g160_a1_dec_best.pth

2. Click Index, then Load model. Architecture: concatenated, FiLM: off.

Required test layout (already present):
  TestProjections/, SourceTestProjections/, DVFs/, Masks/, SourceVolumes/

Raw acquisition (prep_train source): /home/abhishek/Voxel_GAN/VoxelMap_Experiments/runs/CV_P3_V_01/synth_g160_a1_dec/CV_P3_V_01/train
