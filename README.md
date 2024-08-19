# RoWeeder: Unsupervised Weed Mapping through Crop-Row Detection

## Prepare the environment

```bash
conda create -n SSLWeedMap python=3.11
conda activate SSLWeedMap
# Install from environment.yml
conda env update --file environment.yml
```

## Preprocessing

For each field (000, 001, 002, 003, 004)

### Download and extract the dataset

```bash
wget http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Orthomosaic/RedEdge.zip -d dataset/
unzip dataset/RedEdge.zip -d dataset/
```

### Rotate the images

```bash
python3 main.py rotate --root dataset/RedEdge/000 --outdir dataset/rotated_ortho/000 --angle -46 &
python3 main.py rotate --root dataset/RedEdge/001 --outdir dataset/rotated_ortho/001 --angle -48 &
python3 main.py rotate --root dataset/RedEdge/002 --outdir dataset/rotated_ortho/002 --angle -48 &
python3 main.py rotate --root dataset/RedEdge/003 --outdir dataset/rotated_ortho/003 --angle -48 &
python3 main.py rotate --root dataset/RedEdge/004 --outdir dataset/rotated_ortho/004 --angle -48
```

### Patchify the images
```bash
python3 main.py patchify --root dataset/rotated_ortho/000 --outdir dataset/patches/512/000 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/001 --outdir dataset/patches/512/001 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/002 --outdir dataset/patches/512/002 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/003 --outdir dataset/patches/512/003 --patch_size 512 &
python3 main.py patchify --root dataset/rotated_ortho/004 --outdir dataset/patches/512/004 --patch_size 512
```


### Generate the pseudo GT
    
    ```bash
    python3 main.py label --outdir dataset/generated --parameters parameters/row_detect/69023956.yaml
    ```

### Train the RoWeeder Flat model
    
    ```bash
    python3 main.py experiment --parameters=parameters/folds/flat.yaml
