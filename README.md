# PRODIGY_GA_05
# 🎨 Neural Style Transfer using TensorFlow

This project implements **Neural Style Transfer** using a pre-trained **VGG19** network. It blends the **content** of one image with the **style** of another to create a stunning artistic result — all done in pure TensorFlow and Python!

<br/>

## 🧠 What It Does

Given:
- A **content image** (e.g., your photo)
- A **style image** (e.g., Van Gogh painting)

It generates:
- A **stylized image** that keeps the structure of the content but looks like it was painted in the style image's fashion.

<br/>

## 🚀 Features

- Uses **VGG19** pretrained on ImageNet for feature extraction.
- Supports both **Google Colab file upload** and **local file paths**.
- Outputs are saved as `stylized_output.jpg` and displayed using `matplotlib`.

<br/>

## 📦 Required Packages

Install the following dependencies:

```bash
pip install tensorflow pillow matplotlib
