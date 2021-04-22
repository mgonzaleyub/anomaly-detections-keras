# Anomaly Detection in Time Series with Keras
## Objective
In this notebook, I am performing an anomaly detection in time series data with Keras API in Python.
Time series data is any kind of data which varies through time.

The dataset used is S&P500 Daily Prices 1986 - 2018, it can be found in Kaggle through [this link](https://www.kaggle.com/pdquant/sp500-daily-19862018):
Features available are:
- *date*: Date or timestamp is obligatory when processing time series. 
- *close*: Daily Closing Prices for the SPY going from 1986 to 2018. Data not standarized.

The format is similar to the following, in univariate time-series:
Index | date | close
--- | --- | ---
0	| 1986-01-02 |	209.59
1	| 1986-01-03	| 210.88
2	| 1986-01-06 |	210.65
3	| 1986-01-07 | 213.80
4	| 1986-01-08 |	207.97

Because we are predicting a non-discrete value, this is a regression task.

## Preprocessing
Preprocessing pipeline is mandatory in timeseries. As usual in machine learning, we have to
split into training and test datasets, where 80% can be used for training, and 20% remaining for test.

Because *close* feature is not standarized, it's recommended to do it so. Standarizing generally improves performance and implies
a better generalization. I use *Standard Scaler* library from *scikit-learn* for this.

## Train-test splitting
Before feeding a neural network for time series, we have to do one last step: split each dataset into windowed sequences.
> For instance, if we had a time series: [1,2,3,4,5,6,7,8,9] and 3 time-steps for each sequence, we would have: [1,2,3],
> [2,3,4], [3,4,5] and so on.

I've defined 30 timesteps for each sequence.

## Building the neural network
LSTM and RNN are the most accurate approaches for predicting time-series. 
Concretely, an Autoencoder network will be fine for this task.

> An Autoencoder is a neural network which is composed by an encoder network and a decoder network.
> Its main objective is to reduce the input into a low dimensional space and then trying to rebuild into its original shape.
> Anomaly detection is done when Autoencoder can't perform this reconstruction

<img src="https://www.compthree.com/images/blog/ae/ae.png" alt="drawing" width="500"/>

## Architecture chosen
- LSTM layer with 128 neurons and 20% dropout
- RepeatVector layer. This layer repeats the input n times, where n is the timesteps size, 30 in this case.
- LSTM layer with 128 neurons and 20% dropout (same as first layer)
- TimeDistributed output layer, with 1 output neuron. TimeDistributed allows to check every timestep in the sequence, and not only the last value of the seq.
-- I've chosen only one output neuron because its a regression task.

- Optimizer (gradient-descent algorithm) is Adam, useful for time-series.
- Error metric is Mean Absolute Error, useful in regression.

I also included a Early Stopping technique to avoid overfitting, monitoring MAE in validation set when training the Autoencoder.

## Anomaly detection
Anomaly detection implies to choose a threshold value to classify a new sample into normal or abnormal. 
The threshold is chosen by plotting training data (real and predicted with the autoencoder) and choose the highest value.
We are supposing that no anomalies are in training data

So, for new data samples:
- Predict for test dataset
- Apply loss function (MAE) for test predicted and real values.
- Apply threshold
- Filter over-threshold values.

# Certificate
![alt text](https://github.com/mgonzaleyub/anomaly-detections-keras/blob/master/certificate.png "Completion certificate")
