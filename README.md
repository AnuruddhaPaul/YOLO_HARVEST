# YOLO-HARVEST: Hybrid ViT Architecture for Wildlife Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/paper-Ecological%20Informatics-green)](https://github.com/AnuruddhaPaul/YOLO_HARVEST)

Official PyTorch implementation of **"YOLO-HARVEST: A Hybrid ViT Architecture with Locality-Enhanced Attention for Automated Wildlife Species Classification"**


## 🌟 Overview

YOLO-HARVEST is a **two-stage wildlife classification system** designed for automated species identification in camera trap imagery. It combines:
- **YOLOv8** for efficient object detection
- **HARVEST** transformer classifier with novel **Locality-Enhanced Attention (LEA)** mechanisms

The system addresses critical challenges in wildlife monitoring:
- ✅ Extreme class imbalance (6320:1 ratio)
- ✅ Limited training data for rare species
- ✅ Resource-constrained edge deployment
- ✅ Real-time processing requirements for conservation applications

---

## 🚀 Key Features

| Feature | Description | Performance |
|---------|-------------|-------------|
| **High Accuracy** | State-of-the-art wildlife classification | 85.27% (OSU 45 species), 94.74% (Wildlife 6 species) |
| **Lightweight** | Efficient architecture for edge devices | 13M parameters, 52.3 MB (FP32), 14.8 MB (INT8) |
| **Real-Time** | Fast inference for field deployment | 7-8 FPS (Raspberry Pi 5 CPU), 80-125 FPS (GPU) |
| **Robust** | Handles class imbalance and rare species | Median-based oversampling, stratified performance |
| **Modular** | Independent detection/classification stages | Easy component upgrades and debugging |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- CUDA 11.8+ (for GPU support, optional)
- 8GB RAM minimum (16GB recommended)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/AnuruddhaPaul/YOLO_HARVEST.git
cd YOLO_HARVEST

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install YOLOv8
pip install ultralytics
```

### Requirements File
Create `requirements.txt` with:

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
ultralytics>=8.0.0
numpy>=1.24.0
opencv-python>=4.8.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
pillow>=10.0.0
scikit-learn>=1.3.0
onnx>=1.14.0
onnxruntime>=1.16.0
```

---

## 📊 Dataset

### Ohio Small Animals Dataset

Download from **LILA BC** (Labeled Information Library of Alexandria):

```bash
# Download dataset (21,422 images, 45 species)
wget https://lilablobssc.blob.core.windows.net/ohio-small-animals/ohio_small_animals.zip

# Extract
unzip ohio_small_animals.zip -d data/

# Organize structure
mkdir -p data/ohio_small_animals/{train,val,test}
```

**Dataset Structure:**

```
data/
└── ohio_small_animals/
    ├── train/
    │   ├── class_001/
    │   ├── class_002/
    │   └── ...
    ├── val/
    └── test/
```

**Dataset Statistics:**
- **Total Images:** 21,422
- **Species:** 45 North American small mammals
- **Class Imbalance:** 6320:1 (largest:smallest)
- **Median samples/class:** 142 images
- **Extreme minorities (<10 samples):** 8 species

### Wildlife Dataset (Optional)

For balanced dataset evaluation:

```bash
# Download Wildlife dataset (19,000 images, 6 species)
# African megafauna: Cheetah, Elephant, Giraffe, Lion, Rhinoceros, Zebra
```

More details: [LILA BC Datasets](https://lila.science/datasets)

---

## 🏋️ Pretrained Models

| Model | Dataset | Species | Accuracy | F1-Score | Parameters | Size | Download |
|-------|---------|---------|----------|----------|------------|------|----------|
| **YOLO-HARVEST** | OSU Small Animals | 45 | 85.27% | 84.93% | 13.0M | 52.3 MB | [Link](https://github.com/AnuruddhaPaul/YOLO_HARVEST/releases) |
| **YOLO-HARVEST-INT8** | OSU Small Animals | 45 | 84.05% | 83.71% | 13.0M | 14.8 MB | [Link](https://github.com/AnuruddhaPaul/YOLO_HARVEST/releases) |

### Load Pretrained Model

```python
from models.harvest_yolo import YOLO_HARVEST
import torch

# Load model
model = YOLO_HARVEST(num_classes=45, pretrained=True)
model.load_state_dict(torch.load('checkpoints/harvest_osu.pth'))
model.eval()

# For edge deployment (INT8 quantized)
import onnxruntime as ort
session = ort.InferenceSession('checkpoints/harvest_int8.onnx')
```

---

## ⚡ Quick Start

### Single Image Inference

```python
from inference import predict_image

# Predict species
result = predict_image(
    image_path='path/to/wildlife.jpg',
    model_path='checkpoints/harvest_osu.pth',
    device='cuda'  # or 'cpu'
)

print(f"Species: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Bounding Box: {result['bbox']}")
```

### Batch Processing

```bash
python inference.py \
    --input_dir images/ \
    --output_dir predictions/ \
    --weights checkpoints/harvest_osu.pth \
    --batch_size 16 \
    --device cuda
```

---

## 🔧 Training

### Train on Ohio Small Animals

```bash
python train.py \
    --dataset ohio_small_animals \
    --data_path data/ohio_small_animals \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --config configs/harvest_config.yaml \
    --device cuda \
    --num_workers 4
```

### Train with Custom Dataset

```bash
python train.py \
    --data_path /path/to/your/data \
    --num_classes YOUR_NUM_CLASSES \
    --epochs 100 \
    --batch_size 16 \
    --save_dir checkpoints/custom_model
```

### Configuration File

Example `configs/harvest_config.yaml`:

```yaml
model:
  num_classes: 45
  embed_dim: 384
  depth: 12
  num_heads: 6
  patch_size: 16

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 5

augmentation:
  median_oversampling: true
  random_flip: 0.5
  random_rotation: 15
  color_jitter: 0.3
  random_erasing: 0.2

yolo:
  model: yolov8x
  confidence: 0.25
  iou_threshold: 0.45
```

### Training Logs

Monitor training with TensorBoard:

```bash
tensorboard --logdir=runs/
```

---

## 📈 Evaluation

### Full Evaluation

```bash
python evaluate.py \
    --weights checkpoints/harvest_osu.pth \
    --data_path data/ohio_small_animals/test/ \
    --batch_size 32 \
    --save_results results/ \
    --compute_metrics all
```

**Computed Metrics:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR curves
- Confusion matrix (45×45)
- Per-class performance
- Error analysis by failure mode


**Performance:** 7-8 FPS on Raspberry Pi 5 (CPU-only), 520 MB memory, 8.1W power

---

## 🏗️ Model Architecture

### Two-Stage Pipeline

```
Input Image (Camera Trap)
         ↓
┌────────────────────────┐
│  Stage 1: YOLOv8       │
│  Object Detection      │
└────────────────────────┘
         ↓
Detected Bounding Boxes
         ↓
┌────────────────────────┐
│  Stage 2: HARVEST      │
│  Species Classification│
└────────────────────────┘
         ↓
Species Prediction
```

### HARVEST Architecture Components

1. **Self-Patch Tokenizer (SPT)**
   - Preserves spatial locality
   - 16×16 patch size with 2×2 stride
   - Reduces information loss vs. standard ViT

2. **Locality-Enhanced Feature Extractor (LIFE)**
   - Convolutional bottleneck layers
   - Captures local texture patterns
   - Addresses ViT's locality limitations

3. **Locality-Enhanced Attention (LEA)**
   - Modified multi-head self-attention
   - Spatial position encoding
   - Reduced computational complexity

4. **Cross-Level Aggregation (CLA)**
   - Skip connections between layers
   - Multi-scale feature fusion
   - Improves gradient flow

5. **Progressive Classification Heads**
   - Multiple classification stages
   - Hierarchical feature refinement
   - Improved fine-grained discrimination

---

## 📊 Results

### Ohio Small Animals Dataset (45 Species)

| Model | Accuracy | F1-Score | Params | Inference Time |
|-------|----------|----------|--------|----------------|
| **YOLO-HARVEST** | **85.27%** | **84.93%** | 13.0M | 11.2 ms |
| Swin-T | 82.93% | 82.41% | 28.3M | 18.4 ms |
| EfficientNet-B0 | 81.75% | 80.89% | 5.3M | 7.8 ms |
| ResNet50 | 78.45% | 77.83% | 25.6M | 9.4 ms |
| ViT-Base | 73.06% | 71.89% | 86.6M | 22.1 ms |

### Wildlife Dataset (6 Species, Balanced)

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| **YOLO-HARVEST** | **94.74%** | **94.68%** | 0.982 |

### Cross-Dataset Generalization (Zero-Shot)

| Model | CCT Accuracy | Transfer Gap |
|-------|--------------|--------------|
| **YOLO-HARVEST** | **58.24%** | -27.03% |
| Swin-T | 54.18% | -28.75% |
| EfficientNet-B0 | 52.36% | -29.39% |

### Ablation Study Results

| Configuration | Accuracy | Δ Accuracy | Key Finding |
|---------------|----------|------------|-------------|
| Full HARVEST | 85.27% | Baseline | - |
| w/o LIFE | 76.51% | **-8.76%** | Largest contribution |
| w/o Progressive Heads | 77.66% | -7.61% | Multi-stage refinement critical |
| w/o SPT | 78.34% | -6.93% | Spatial locality preservation important |
| w/o LEA | 79.18% | -6.09% | Enhanced attention beneficial |
| w/o CLA | 79.49% | -5.78% | Skip connections improve flow |

### Performance by Sample Size

| Sample Range | # Classes | Avg Accuracy | Challenge Level |
|--------------|-----------|--------------|-----------------|
| ≥1000 samples | 6 (13.3%) | 88.32% | Low |
| 100-1000 | 18 (40.0%) | 84.67% | Medium |
| 10-100 | 13 (28.9%) | 76.42% | High |
| <10 samples | 8 (17.8%) | 62.15% | Extreme |

---

## 📝 Citation

If you use YOLO-HARVEST in your research, please cite:

```bibtex
@article{paul2025yoloharvest,
  title={YOLO-HARVEST: A Hybrid ViT Architecture with Locality-Enhanced 
         Attention for Automated Wildlife Species Classification},
  author={Paul, Anuruddha and Raj, Rishi and Gourisaria, Mahendra Kumar 
          and Jha, Amitkumar V. and Bizon, Nicu},
  journal={Ecological Informatics},
  year={2025},
  publisher={Elsevier},
  note={Manuscript ID: ECOINF-D-25-03227}
}
```

**Paper:** [Link to published paper]  
**ArXiv:** [Preprint link if available]

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2025 Anuruddha Paul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for Contribution:**
- Additional dataset support
- Model architecture improvements
- Edge deployment optimizations
- Bug fixes and documentation

---

## 📧 Contact

**Authors:**
- **Anuruddha Paul** - [GitHub](https://github.com/AnuruddhaPaul)
- **Rishi Raj** - Information Systems, IIM Visakhapatnam
- **Mahendra Kumar Gourisaria** (Corresponding Author) - [mkgourisaria2010@gmail.com](mailto:mkgourisaria2010@gmail.com)
- **Amitkumar V. Jha** - KIIT Deemed to be University
- **Nicu Bizon** (Corresponding Author) - [nicu.bizon1402@upb.ro](mailto:nicu.bizon1402@upb.ro)

**Project Link:** [https://github.com/AnuruddhaPaul/YOLO_HARVEST](https://github.com/AnuruddhaPaul/YOLO_HARVEST)

---

## 🙏 Acknowledgments

- Dataset provided by **LILA BC** (Labeled Information Library of Alexandria: Biology and Conservation)
- Research supported by the **PubArt program** of the National University of Science and Technology POLITEHNICA Bucharest
- Built with [PyTorch](https://pytorch.org/), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), and [TIMM](https://github.com/rwightman/pytorch-image-models)
- Compliance with open science principles following **Huettmann & Arhonditsis (2023)** guidelines

---

## 📚 References

Key references for reproducibility and methodology:

```bibtex
@article{huettmann2023open,
  title={A multi-year open access data analysis of a 'niche swap' by 
         the Short-billed Gull (\textit{Larus brachyrhynchus}) in 
         urban-suburban Fairbanks, Alaska during late spring and summer},
  author={Huettmann, Falk and Arhonditsis, George B.},
  journal={Global Ecology and Conservation},
  volume={42},
  pages={e02391},
  year={2023},
  publisher={Elsevier},
  note={Open science data sharing guidelines}
}

@misc{osu_dataset,
  title={{Ohio Small Animals Camera Trap Dataset}},
  author={{LILA BC}},
  howpublished={Labeled Information Library of Alexandria: Biology and Conservation},
  year={2023},
  url={https://lila.science/datasets/ohio-small-animals},
  note={21,422 images of 45 species}
}
```

---

**⭐ If you find this work useful, please consider starring the repository!**

---

*Last Updated: November 2025*
