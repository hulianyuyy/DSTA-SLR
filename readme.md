# [Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition](https://arxiv.org/pdf/2403.12519.pdf)
This work focuses on skeleton-aware sign language recognition (SLR) which receives a series of skeleton points to classify the classes of sign language. Compared to RGB-based inputs, it consumes <1/3 computations and achieve >2× inference speed. This work proposes a dynamic spatial-temporal aggregation network for skeleton-aware sign language recognition. It achieves new state-of-the-art performance on four datasets including NMFs-CSL, SLR500, MSASL and WLASL and outperforms previous methods by a large margin.

## Data preparation
The preprocessed skeleton data for NMFs-CSL, SLR500, MSASL and WLASL datasets are provided [here](https://mega.nz/folder/JytAARrA#76Ug-Khu-8Eskmrw1HhMCQ). Please be sure to follow their rules and agreements when using the preprocessed data.

For datasets (WLASL100, WLASL300, WLASL1000, WLASL2000, MLASL100, MLASL200, MLASL500, MLASL1000, SLR500, NMFs-CSL) used to train or test our model, first create a soft link. For example, for WLASL2000:
```
ln -s path_to_your_WLASL2000/WLASL2000/ ./data/WLASL2000
```

## Pretrained models
We provide the pretrained weight for our model on the WLASL2000 dataset to validate its performance in [./pretrained_models](./pretrained_models)

## Installation
To install necessary packages, run this command. 
```bash
pip install -r requirements.txt
```

## Training and testing:
Conduct the following commands: 
```
mkdir save_models
```
### Training

```
python -u main.py --config config/train.yaml --device your_device_id
```

### Testing:
```
python -u main.py --config config/test.yaml --device your_device_id
```

To test your model with pretrained weights, you may modify the line 52 in [./config/test.yaml](./config/test.yaml) to path of your pretrained weight.

Update: There seems to be an error that loading pretrained models doesn't give correct inference results. This doesn't affect the normal training procedure.

### Ensembling 
To obtain the final reults reported in the paper by perform multi-stream fusing based on the joint, bone, joint-motion and bone-motion streams, you should first set the `bone_stream` and `motion_stream` in line 16 & 17 and line 26 & 27 in the `./config/train.yaml` as [True, True], [True, False], [False, True] and [False, False], respectively, to run four times obtain the results of different streams.

To perform multi-stream fusing, modify the path to your result file in the [./ensemble/ensemble.py](./ensemble/ensemble.py) in lines 12-18 for the four streams, and select the fusing weights from line 21-30 according to your dataset. The `pkl` file is located in your `work_dir`. Then conduct `python ./ensemble/ensemble.py`.

## Acknowledgements

This code is based on [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2) and [SLGTformer](https://github.com/neilsong/SLGTformer). Many thanks for the authors for open sourcing their code.
