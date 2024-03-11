# Self-Supervised-Learning-for-Precision-Agriculture

## Preprocessing

For each field (000, 001, 002, 003, 004)

### 1. [Download the dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap#orthomosaic)

### Rotate the images

```bash

# Rotate the images
python3 main.py rotate --root dataset/ortho/000 --outdir dataset/rotated_ortho/000 --angle -48 & 
python3 main.py rotate --root dataset/ortho/001 --outdir dataset/rotated_ortho/001 --angle -48 &
python3 main.py rotate --root dataset/ortho/002 --outdir dataset/rotated_ortho/002 --angle -48 &
python3 main.py rotate --root dataset/ortho/003 --outdir dataset/rotated_ortho/003 --angle -48 &
python3 main.py rotate --root dataset/ortho/004 --outdir dataset/rotated_ortho/004 --angle -48

# Patchify the images
python3 main.py patchify --root dataset/rotated_ortho/000 --outdir dataset/patches/512/000 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/001 --outdir dataset/patches/512/001 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/002 --outdir dataset/patches/512/002 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/003 --outdir dataset/patches/512/003 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/004 --outdir dataset/patches/512/004 --patch_size 512