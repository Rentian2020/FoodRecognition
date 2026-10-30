# FoodRecognition

Food image classification using **MobileNetV3-Small** and transfer learning with the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). This project explores lightweight food recognition for an RGB-camera-based dietary-intake prototype, comparing a 10-class baseline, partial fine-tuning, and full 101-class classification.

## Models and results

| Checkpoint | Classes | Training approach | Reported performance |
|---|---:|---|---|
| `models/mobilenetv3_food_v2.pth` | 10 | ImageNet-pretrained MobileNetV3-Small; frozen feature extractor; trained classification head | **84.00% test accuracy** |
| `models/mobilenetv3_food_v3.pth` | 10 | MobileNetV3-Small; partial fine-tuning of the last three feature blocks | **85.00% test accuracy** |
| `models/mobilenetv3_food101.pth` | 101 | ImageNet-pretrained MobileNetV3-Small; frozen feature extractor; full Food-101 classification | **59.53% validation accuracy** at epoch 4 |

**Important:** The 101-class figure is *validation* accuracy, not a held-out test result. The 10-class and 101-class results are not directly comparable because the classification tasks have different numbers of categories.

An earlier 10-class prototype achieved 83.20% test accuracy after three epochs; its checkpoint is not one of the three checkpoints listed above.

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) contains **101,000 images** across **101 food categories**, with 750 training images and 250 test images per class.

- **10-class experiments:** 6,000 training / 1,500 validation / 2,500 test images.
- **101-class experiment:** 60,600 training / 15,150 validation images; 25,250 official test images reserved.
- Input images are resized to **224 × 224** for MobileNetV3-Small.

The dataset is **not included** in this repository. Download it separately from the link above.

## Training

The project uses PyTorch and ImageNet-pretrained MobileNetV3-Small. The experiments investigate transfer learning, fine-tuning, regularization, and the effect of scaling from 10 to 101 food categories.

For the full 101-class experiment, the configuration included a batch size of 64, Adam optimizer, learning rate of `1e-3`, weight decay of `1e-4`, and early stopping. Training and validation metrics are recorded in the `results/` CSV files.

## Repository structure

```text
FoodRecognition/
├── train.py                     # Training experiments
├── finetune.py                  # Fine-tuning experiment
├── train_101.py                 # Full Food-101 training
├── evaluate.py                  # Model evaluation
├── predict.py                   # Single-image prediction
├── bigMac.jpg                   # Example input image
├── models/
│   ├── mobilenetv3_food_v2.pth
│   ├── mobilenetv3_food_v3.pth
│   └── mobilenetv3_food101.pth
├── results/
│   ├── finetuning_history_v3.csv
│   ├── training_history_v2.csv
│   └── training_history_101.csv
└── .gitignore
```

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/Rentian2020/FoodRecognition.git
   cd FoodRecognition
   ```

2. Set up a Python environment and install the dependencies used by the scripts, including PyTorch, torchvision, Pillow, and other imports required by your local version.

3. Download and extract Food-101 into the dataset location expected by the training scripts if you want to retrain or evaluate on the dataset.

4. Review `predict.py` to select the checkpoint and input image, then run:

   ```bash
   python predict.py
   ```

   The exact checkpoint selection and image path depend on the configuration in `predict.py`.

## Limitations and next steps

- The 10-class models only recognize their selected food categories; they are **not** general 101-class classifiers.
- The 101-class checkpoint requires further evaluation on the reserved test split.
- Food classification alone does not estimate portion size or calories.
- Future work includes improved augmentation, better generalization across all 101 classes, and integration with RGB-camera image capture.

## Project context

Developed as the food-recognition component of a wearable dietary-intake monitoring prototype. The current repository focuses on **model training and image classification**, not a complete deployed dietary-monitoring system.
