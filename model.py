import warnings
warnings.simplefilter(
    action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import sklearn.preprocessing as sk

tf.compat.v1.disable_eager_execution()


class Seq2SeqModel:

    def __init__(self, input_shape, lstm_dim, output_size):
        self.encoder = []
        self.decoder = []
        self.decoder_dense = []
        self.encoder_states = []
        self.encoder_input = []
        self.decoder_input = []
        self.lstm_dim = lstm_dim
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = self.init_training_model()

    def init_training_model(self):
        model = Sequential([
            LSTM(self.lstm_dim, return_sequences=True),
            Dense(self.output_size)
        ])

        optimizer = keras.optimizers.SGD(
            lr=0.0005, momentum=0.8, nesterov=True)
        loss = tf.keras.losses.MeanAbsoluteError()

        model.compile(optimizer=optimizer, loss=loss)

        return model

    def predict_seq(self, input_seq):

        return self.model.predict(input_seq)[0]


if __name__ == "__main__":
    scaler = sk.MinMaxScaler()
    in_data = np.array([[1], [2], [3], [4], [5], [6]])
    out_data = np.array([[2], [3], [4], [5], [6], [7]])
    target_data = np.array([[2], [3], [4], [5]])
    scaler.fit(in_data)
    in_data = scaler.transform(in_data)
    out_data = scaler.transform(out_data)
    target_data = scaler.transform(target_data)
    in_data = np.expand_dims(in_data, axis=0)
    out_data = np.expand_dims(out_data, axis=0)
    target_data = np.expand_dims(target_data, axis=0)

    a = Seq2SeqModel(input_shape=(None, 1),
                     lstm_dim=200, output_size=1)
    #tf.keras.utils.plot_model(a.model, "test.png")
    a.model.fit(x=in_data, y=out_data, epochs=2000)

    v = a.predict_seq(target_data)
    print(v)
    print(scaler.inverse_transform(v))
