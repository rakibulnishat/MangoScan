Mango leaf disease classification using transfer learning with dual XAI validation — training notebooks and web application

# MangoScan — Mango Leaf Disease Classifier

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/nishaatt/mango-disease-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

## Demo Preview
![MangoScan Demo](assets/demo.png)

## Overview

This repository contains the training notebooks and web application source code 
for a comparative analysis of five transfer learning architectures for mango leaf 
disease classification under balanced and imbalanced dataset conditions, with dual 
Explainable AI validation using Grad-CAM and SHAP.

**Paper:** Under review — citation will be added upon publication.

## Live Demo

Try MangoScan directly in your browser — no installation required:
**https://huggingface.co/spaces/nishaatt/mango-disease-detector**

## Repository Structure
MangoScan/
├── BALANCED/        # Training on MangoLeafBD balanced (500/class)
├── IMBALANCED/      # Training on MangoLeafBD imbalanced (206-360/class)
└── app/             # MangoScan Gradio web application

## Models

Five pretrained transfer learning architectures evaluated:

| Model | Parameters | Balanced Acc. | Imbalanced Acc. |
|---|---|---|---|
| ResNet50 | 25.6M | 100.00% | 98.73% |
| VGG16 | 138.4M | 100.00% | 97.46% |
| EfficientNetB3 | 12.3M | 100.00% | 98.31% |
| InceptionV3 | 23.9M | 99.75% | 96.19% |
| MobileNetV2 | 3.4M | 99.50% | 95.76% |

## Dataset

MangoLeafBD — 8 disease classes, publicly available on Kaggle:
https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

## Model Weights

Trained model weights are not stored in this repository.
The deployed ResNet50 model is available via the Hugging Face Space above.

## Requirements

See `requirements.txt` for the full dependency list.
Main dependencies: TensorFlow 2.19.0, Gradio 5.9.1, SHAP, OpenCV, Kagglehub.

## Citation

Paper under review. Citation will be updated upon acceptance.

## License

MIT License — see [LICENSE](LICENSE) for details.
