# Error Analysis of Floating-Point Algorithms

This repository contains the implementation and experiments from my MSc dissertation:
**"Error Analysis of Floating-Point Algorithms to Compute Inner Products and the QR Factorization, with an Application to Least Squares Problems"**.

## 🔍 Overview

This project focuses on:
- Rounding errors in floating-point systems
- Accuracy analysis of inner product calculations
- QR decomposition via Householder reflections (element-wise and block-wise)
- Mixed-precision computation effects
- Error bounds for least-squares solutions

## 🧪 Structure

- `src/`: Core Python modules for computation
- `results/`: Output logs and figures
- `docs/`: Analytical summaries and algorithmic derivations

## 🧠 Key Concepts

- IEEE-754 floating-point model
- Probabilistic error bounds
- Mixed-precision fused multiply-add (FMA)
- Log-log regression on relative error bounds

## 📊 Results Preview

All experiments were conducted with `NumPy` via Google Colab, supporting the theoretical claims in the dissertation.

## 📄 License

MIT License
