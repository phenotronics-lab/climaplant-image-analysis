# CLIMAPLANT image analysis

Image analysis code for CLIMAPLANT (CLImate-controlled iMAging for PLANTs), a growth chamber for
Arabidopsis thaliana with chlorophyll fluorescence, thermal infrared and RGB imaging.

- `scripts/`: rosette segmentation (U-Net), leaf tracking (YOLO11l-seg + LapTrack) and
  registration of the masks onto the RGB and infrared images
- `training/`: training and evaluation of the two models

Data and model weights : https://doi.org/10.5281/zenodo.24038899 (active after publication).

## Installation

```
conda env create -f environment.yml
conda activate climaplant
```

Tested on Windows with an NVIDIA GPU (8 GB).

## Data

Extract `example.zip`, `calibration.zip`, `unet.zip` and `yolo.zip` in `data/`, and `models.zip`
in the repository folder:

```
data/
  example/       images of one plant (POS12): FLUO-NoSat, FLUO-ApSat, FLUO-AvSat, VIS-Nat, IR-Nat
  calibration/   calibration points
  unet/          U-Net images and masks (train, val, test)
  yolo/          YOLO images and labels (train, val)
models/
  unet_best_model.pth
  yolo11l_seg_best.pt
```

Image names: `<id>-<type>-<request date>-<request time>-<capture date>-<capture time>-<position>-...tiff`
(dates `ddmmyy`, times `HHMMSS`).

## Usage

Run from the repository folder. Parameters are at the top of each script, outputs go to `results/`.

```
python scripts/segment_rosettes.py
python scripts/register_masks.py
python scripts/track_leaves.py
```

- `segment_rosettes.py`: rosette masks from the NoSat fluorescence images.
- `register_masks.py`: projects the masks onto the RGB and infrared images with a homography
  (`cv2.findHomography`, `cv2.warpPerspective`). Needs the masks from `segment_rosettes.py`.
- `track_leaves.py`: leaf segmentation with YOLO and tracking with LapTrack.

The outputs of these scripts for the example plant are in `results.zip` on Zenodo (extract it in the repository folder).

## Evaluation

```
python training/unet/evaluate_unet.py
python training/yolo/evaluate_yolo.py
```

- U-Net, 31 test images: Dice 0.971, IoU 0.945 (before post-processing)
- YOLO11l-seg, 61 validation images: mask mAP@0.5:0.95 0.82, mask mAP@0.5 0.97

## Training

```
python training/unet/train_unet.py
python training/yolo/train_yolo11_seg.py
```

Results in the article come from the weights in `models/`. Full training takes several hours.
Pretrained weights are downloaded on the first run.

## License

MIT
