# Bitcoin Time Series Forecasting

This repository contains a Jupyter notebook for forecasting Bitcoin prices using time series analysis techniques. The notebook employs both Long Short-Term Memory (LSTM) neural networks and ARIMA models to predict future Bitcoin prices based on historical data.

## Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [LSTM Model](#lstm-model)
  - [ARIMA Model](#arima-model)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Introduction

Bitcoin, the first and most well-known cryptocurrency, has a volatile price history. This notebook explores forecasting Bitcoin prices using two different approaches: LSTM neural networks and ARIMA models. The goal is to compare the performance of these models and determine their accuracy in predicting future prices.

## Dataset

The dataset used in this notebook consists of historical Bitcoin market data at 1-minute intervals from various exchanges. It includes Open, High, Low, Close prices, Volume in BTC and the indicated currency, and the weighted Bitcoin price.

## Data Preprocessing

Before modeling, the data undergoes several preprocessing steps, including:

- Converting timestamps to datetime format.
- Resampling the data to a daily frequency.
- Handling missing values and outliers.

## Modeling

### LSTM Model

A Long Short-Term Memory (LSTM) neural network is implemented using Keras. The model architecture includes:

- An LSTM layer with 128 units and ReLU activation.
- Dropout layers to prevent overfitting.
- Multiple LSTM layers with varying units.
- Dense layers for the final output.

The model is trained for 100 epochs with a batch size of 50.

### ARIMA Model

An ARIMA (AutoRegressive Integrated Moving Average) model is also implemented for comparison. The steps include:

- Initial approximation of parameters.
- Model selection based on the Akaike Information Criterion (AIC).
- Forecasting future values using the best model.

## Model Evaluation

The models are evaluated using the following metrics:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

These metrics are calculated for both the LSTM and ARIMA models to compare their performance.

## Results

The results section includes plots of the actual vs. predicted prices for both models. Additionally, a bar chart compares the MSE, MAE, and RMSE of the two models.

## Usage

To run the notebook, follow these steps:

1. Clone this repository.
2. Install the required packages (see [Requirements](#requirements)).
3. Open the notebook `Bitcoin_Time_Series_Forecasting.ipynb` in Jupyter.
4. Execute the cells in the notebook to preprocess the data, train the models, and evaluate their performance.

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Keras
- TensorFlow
- Statsmodels

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib scikit-learn keras tensorflow statsmodels

## LSTM

## Arima
