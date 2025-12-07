# Forecasting GitHub Repository Stars
## ARMA, RNN, and 1D CNN Models

**Author:** Aditya Peketi

---

## 1. Introduction
This report presents forecasting experiments conducted on two GitHub repositories: **`facebook-react`** and **`pallets-flask`**. I have visually check the csv file and have picked these two and I have observed that every repo saturates after 4000 stars so I have cleaned the data.

The objective is to model and predict future **cumulative star counts** using three approaches:
1.  **Classical statistical model:** ARMA.
2.  **Deep learning models:** Simple RNN and 1D CNN.

---

## 2. Data Preparation
Each dataset contains a timestamp column and a cumulative star count. Data preprocessing included:

* **Time Series Preparation:** Parsing timestamps as `datetime` objects and sorting chronologically.
* **Train-Test Split:** Using an **80–20 chronological split** for training and testing.
* **Normalization:** Scaling only the training portion with **MinMaxScaler**; test data reused the same fit.
* **Supervised Learning Format:** Converting the time series into supervised learning format using a **sliding window of 20 steps** (the lookback window):
    * Input Sequence: $X_t = [s_{t-19}, \ldots, s_t]$
    * Target Value: $y_{t+1} = s_{t+1}$

> ARMA models used the raw cumulative training series directly without scaling.

---

## 3. Forecasting Models

### 3.1 ARMA Model
A classical **ARMA(2,2)** model was fitted using `statsmodels`.

It performs **one-step-ahead recursive forecasting** over the test region by refitting the model at each step (Rolling Forecast Origin). Although cumulative time series are typically **nonstationary** (they exhibit a clear trend), ARMA is included for baseline comparison.

### 3.2 Simple RNN
The **RNN** architecture consisted of:
* Two recurrent layers with **64 hidden units**.
* A fully connected output layer.

It receives a $20 \times 1$ sequence and predicts the next cumulative star value. Training used the Adam optimizer for 50 epochs. RNNs can model temporal dependencies but may struggle with long-range patterns due to **vanishing gradients**.

### 3.3 1D CNN Forecaster
The **1D CNN** architecture consisted of:
* **Conv1D** layer with 64 filters.
* **MaxPooling** layer.
* **Conv1D** layer with 128 filters.
* **MaxPooling** layer.
* **Dense layers** (64 units then 1 output).

This model extracts **local temporal patterns** efficiently and trains significantly faster than the RNN. CNNs handle nonlinear dynamics well and often generalize better on cumulative series.

---

## 4. Quantitative Results
Each model was evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** over the test region. A typical results table is shown below:

| Repository & Domain            | Model | MAE       | RMSE      |
|--------------------------------|-------|-----------|-----------|
| facebook-react (Cumulative)    | ARMA  | 14.3824   | 20.7294   |
|                                | RNN   | 335.7124  | 340.6105  |
|                                | CNN   | 57.6467   | 72.5799   |
| facebook-react (Incremental)   | ARMA  | 8.2576    | 12.2273   |
|                                | RNN   | 8.7726    | 12.2831   |
|                                | CNN   | 7.3995    | 12.4193   |
| pallets-flask (Cumulative)     | ARMA  | 0.8170    | 1.1228    |
|                                | RNN   | 11.4477   | 11.6230   |
|                                | CNN   | 11.0149   | 11.3535   |
| pallets-flask (Incremental)    | ARMA  | 0.7728    | 0.9163    |
|                                | RNN   | 1.0107    | 1.1610    |
|                                | CNN   | 1.3141    | 1.5169    |


---

Please look at the notebook for the results.