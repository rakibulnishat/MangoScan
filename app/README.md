---
title: MangoScan — Mango Leaf Disease Detector
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
---

# 🌿 MangoScan — Mango Leaf Disease Detector

A deep learning app that classifies mango leaf diseases from photos using a fine-tuned **ResNet50** model.

## Detectable Conditions

| Class | Severity |
|---|---|
| Anthracnose | High |
| Bacterial Canker | High |
| Cutting Weevil | Medium |
| Die Back | High |
| Gall Midge | Medium |
| Healthy | — |
| Powdery Mildew | Medium |
| Sooty Mould | Low |

## Model Details

- **Architecture:** ResNet50 (ImageNet pretrained, fine-tuned)
- **Dataset:** [Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset) — 4,000 balanced images, 8 classes
- **Training:** 2-phase (head training + last-block fine-tuning)
- **Test Accuracy:** 100%
- **Explainability:** Grad-CAM + SHAP analysis included in research

## Author

Rakibul Hassan (Nishat) · Mechatronics Engineering · RUET, Bangladesh
