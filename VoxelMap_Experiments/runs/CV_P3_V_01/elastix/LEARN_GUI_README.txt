LEARN-GUI DVF Visualisation
============================

Run folder: /home/abhishek/Voxel_GAN/VoxelMap_Experiments/runs/elastix_CV_P3_V_01

1. Open LEARN-GUI-Python and load this folder as the active run, OR
   Menu → DVF Visualisation and set:
   Patient folder: ModelTraining/test/CV_P3_V_01
   Model weights:  Models/weights_concatenated_nofilm_elastix_best.pth

2. Click Index, then Load model. Architecture: concatenated, FiLM: off.

Required test layout (already present):
  TestProjections/, SourceTestProjections/, DVFs/, Masks/, SourceVolumes/

Raw acquisition (prep_train source): /home/abhishek/Voxel_GAN/VoxelMap_Experiments/runs/elastix_CV_P3_V_01/CV_P3_V_01/train
