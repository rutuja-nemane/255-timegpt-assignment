# TimeGPT, Tabula, and RelBench Project

This repository contains Colab notebooks showcasing machine learning tasks using TimeGPT, Tabula, and RelBench. Each notebook demonstrates various capabilities, including forecasting, anomaly detection, synthetic data generation, and tabular prediction. A video presentation walks through each notebook’s purpose, code, and outputs.

## Table of Contents
- [TimeGPT](#timegpt)
  - [Multivariate & Long Horizon Forecasting]([#multivariate--long-horizon-forecasting](https://colab.research.google.com/drive/182RoYBLd4BXIr4Z1PcmVqJjcUAuwi65E?usp=sharing))
  - [Fine-tuning with Custom Data](#fine-tuning-with-custom-data)
  - [Anomaly Detection](#anomaly-detection)
  - [Energy Forecasting](#energy-forecasting)
  - [Bitcoin Price Prediction](#bitcoin-price-prediction)
- [Tabular](#tabular)
  - [Synthetic Data Generation](#synthetic-data-generation)
  - [Zero-shot Inference](#zero-shot-inference)
- [RDL and RelBench](#rdl-and-relbench)
  - [GNN-based Model Training for Tabular Prediction](#gnn-based-model-training-for-tabular-prediction)

---

## TimeGPT

### Multivariate & Long Horizon Forecasting
- **Notebook**: `Multivariate_long_horizon.ipynb`
- **Description**: Forecast multiple time series over long horizons to understand relationships across variables.
- **Steps**: Data preparation, model configuration, forecasting, and results visualization.

### Fine-tuning with Custom Data
- **Notebook**: `finetune.ipynb`
- **Description**: Fine-tune TimeGPT for specific time series tasks to enhance accuracy.
- **Steps**: Load custom data, run fine-tuning, and evaluate predictions.

### Anomaly Detection
- **Notebook**: `Anomaly_detection.ipynb`
- **Description**: Detect anomalies in time series data to identify significant deviations.
- **Steps**: Dataset preparation, configure detection, and visualize anomalies.

### Energy Forecasting
- **Notebook**: `Energy_forecasting.ipynb`
- **Description**: Forecast energy demand to manage resources efficiently.
- **Steps**: Load energy data, configure forecasting model, and analyze results.

### Bitcoin Price Prediction
- **Notebook**: `Bitcoin_forecasting.ipynb`
- **Description**: Predict Bitcoin prices, demonstrating TimeGPT’s capability in volatile data handling.
- **Steps**: Import Bitcoin data, train the model, and visualize forecasts.

## Tabular

### Synthetic Data Generation
- **Notebook**: `synthetic_data_for_a_real_data_set.ipynb`
- **Description**: Generate synthetic data that resembles real datasets to aid in model training.
- **Steps**: Load real data, create synthetic dataset, and compare for feature consistency.

### Zero-shot Inference
- **Notebook**: `zero_shot_inference.ipynb`
- **Description**: Perform zero-shot inference, showcasing Tabula’s predictive capability without prior dataset training.
- **Steps**: Model setup, data input, and analyze model generalization.

## RDL and RelBench

### GNN-based Model Training for Tabular Prediction
- **Notebook**: `train_model.ipynb`
- **Description**: Train a GNN model using RelBench for relational data predictions.
- **Steps**: Prepare tabular data, train GNN, and evaluate performance metrics.
