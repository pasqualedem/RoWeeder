# Self-Supervised-Learning-for-Precision-Agriculture

## Preprocessing

For each field (000, 001, 002, 003, 004)

### 1. [Download the dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap#orthomosaic)

### Rotate the images

```bash

# Rotate the images
python3 main.py rotate --root ../Datasets/WeedMap/ortho/<field_code> --outdir dataset/ortho/<field_code> --angle -48

