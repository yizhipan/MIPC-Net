# DA-TransUnet
MIPC-Net: A Mutual Inclusion Mechanism for Precise Boundary Segmentation in Medical Images (https://arxiv.org/abs/2404.08201)

### 1.Prepare pre-trained ViT models
* [Get models and training parameters in this link](https://drive.google.com/drive/folders/1UqIEPcohjIZdpT5bIc0NPcxkvI8i4ily): R50-ViT-B_16,At the same time, the parameter file (.pth) in the paper is also stored.(You can download and compress it, put it into the model file and rename it TU_Synapse224, and then use the test code (python test.py --dataset Synapse --vit_name R50-ViT-B_16) to get the test results.)

### 2.Prepare data
Please use the [preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) for research purposes.

### 3.Environment
Please prepare an environment with python=3.7(conda create -n envir python=3.7.12), and then use the command "pip install -r requirements.txt" for the dependencies.

### 4.Train/Test
Run the train script on synapse dataset. The batch size can be reduced to 12 or 16 to save memory(please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference 
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [TransUnet](https://github.com/Beckschen/TransUNet)

## Citation

