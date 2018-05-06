import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Flatten, BatchNormalization, Dense, Dropout
from keras.constraints import max_norm
from keras import regularizers

import keras.backend as K

import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from scipy.io import loadmat

data_dir = 'E:/brains/eeg/problem_solving/'

def load_data(data_dir):
    time_series, labels = [], []

    for filename in os.listdir(data_dir):
        # print(filename)
        if 'uvjet1' in filename:

            if 'nen' in filename:
                labels.append(0)
            else:
                labels.append(1)

            eeg_data = loadmat(data_dir + filename)

            for key in eeg_data.keys():
                if not '__' in key:
                    time_series.append(eeg_data[key])

    print('labels:', len(labels))
    print('time series:', len(time_series))

    return time_series, np.asarray(labels)

def preprocess_timeseries(data):
    max_length = 0
    for sequence in data:
        if max_length < sequence.shape[-1]:
            max_length = sequence.shape[-1]

    preprocessed = np.zeros((len(data), 64, max_length))
    # print(preprocessed.shape)

    for i, sequence in enumerate(data):
        pad_amount = max_length - sequence.shape[-1]
        new_sequence = np.pad(sequence, ((0,0), (0, pad_amount)), mode='constant')

        new_sequence /= np.max(np.abs(new_sequence))
        # print(new_sequence.shape)
        preprocessed[i, :, :] = new_sequence

    return preprocessed

def sequential_model(n_channels, n_timepoints):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_channels, n_timepoints)))
    model.add(LSTM(128, dropout=0.5, recurrent_constraint=max_norm(2.), return_sequences=True))
    model.add(LSTM(128, dropout=0.5, recurrent_constraint=max_norm(2.), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    n_epochs = 10
    n_folds = 5

    data, labels = load_data(data_dir)
    data = preprocess_timeseries(data)

    print('data shape', data.shape)

    n_samples = data.shape[0]
    n_channels = data.shape[1]
    n_timepoints = data.shape[2]

    all_test_truth, all_test_probabilities, all_validation_truth, all_validation_predictions = [], [], [], []

    sss = StratifiedKFold(n_splits=n_folds, random_state=42)
    for train_indices, other_indices in sss.split(data, labels):
        validation_indices = other_indices[::2]
        test_indices = other_indices[1::2]

        model = sequential_model(n_channels, n_timepoints)
        model.summary()

        results = np.zeros((n_epochs, len(train_indices), 1))

        for epoch_idx in range(n_epochs):
            np.random.shuffle(train_indices)

            for i, index in enumerate(train_indices):
                x = data[index, ...][np.newaxis, ...]
                y = labels[index, ...][np.newaxis, ...]

                # print('data sample:', x)
                # print('label:', y)
                metrics = model.train_on_batch(x, y)
                # print(metrics)

                results[epoch_idx, i, 0] = metrics[1]

            print('Epoch', str(epoch_idx + 1), np.mean(results[epoch_idx, :, 0])*100, '% accuracy')

        test_predictions, test_truth = [], []
        validation_predictions, validation_truth = [], []
        for index in validation_indices:
            x = data[index, ...][np.newaxis, ...]
            y = labels[index, ...][np.newaxis, ...]

            prediction = model.predict_on_batch(x)

            if prediction > 0.5:
                validation_predictions.append(1)
            else:
                validation_predictions.append(0)

            validation_truth.append(y)
            all_validation_truth.append(y)
            all_validation_predictions.append(prediction)

        for index in test_indices:
            x = data[index, ...][np.newaxis, ...]
            y = labels[index, ...][np.newaxis, ...]

            prediction = model.predict_on_batch(x)

            print('truth:', y, 'predicted:', prediction)

            if prediction > 0.5:
                test_predictions.append(1)
            else:
                test_predictions.append(0)

            test_truth.append(y)
            all_test_truth.append(y)
            all_test_probabilities.append(prediction)

        test_score = accuracy_score(test_truth, test_predictions)
        validation_score = accuracy_score(validation_truth, validation_predictions)
        print('validation accuracy:', validation_score*100, '%')
        print('test accuracy:', test_score*100, '%')

        #TODO: early stopping, plotting results, iterating through dropping channels for interpretability

        K.clear_session()
