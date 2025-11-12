# 🧠 Conformal Prediction Implementations

This repository contains concise, educational implementations of **Conformal Prediction (CP)** using both **NumPy**/**SciPY** and **TorchCP**.  
It illustrates how to construct reliable, **distribution-free prediction sets** that quantify model uncertainty and ensure coverage guarantees.

---

## 🧭 Overview

This project demonstrates the practical and theoretical aspects of **distribution-free uncertainty quantification** through two complementary implementations:

- 🧮 **NumPy Implementation** — a fully transparent, from-scratch version for educational and research purposes using only NumPy and SciPy.  
- ⚙️ **TorchCP Implementation** — deep learning–based examples leveraging PyTorch and the open-source [TorchCP](https://github.com/ml-stat-Sustech/TorchCP) library.

These implementations aim to highlight:
- How conformal prediction ensures valid **coverage calibration**
- Construction of **prediction sets** under minimal assumptions
- Comparison between classical and neural methods

---



## 📂 Repository Structure

```text
Conformal-Prediction-Implementations/
├── src/
│   ├── pure_python/      # NumPy / SciPy from-scratch implementations
│   └── torch_cp/         # PyTorch + TorchCP implementations
│
└── notebooks/            # Jupyter notebooks with demos & visualizations


📚 References

Angelopoulos, A. N., & Bates, S. (2022).
A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
arXiv preprint arXiv:2107.07511


