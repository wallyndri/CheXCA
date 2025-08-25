# CheXCA: Chest X-ray Classification with Cross-Attention and GAT

## 📌 Overview
CheXCA is a deep learning model designed for **multi-label chest X-ray classification**.  
It addresses:
- Overlapping pathologies
- Extreme class imbalance
- Calibration of confidence estimates

## ⚙️ Components
- **ConvNeXt-Base** as backbone
- **Class-Token Cross-Attention**
- **Graph Attention Network (GAT)**
- **Focal Loss + Differentiable ECE Loss**

## 🚀 How to Run
1. Clone repo:
   ```bash
   git clone https://github.com/your-username/CheXCA.git
   cd CheXCA