import warnings
warnings.simplefilter(
    action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import Model
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
        # self.init_inference_model()
        a = np.zeros(shape=(1, 1, 514))
        # self.predict_seq(a)

    def init_training_model(self):
        # Init encoder
        self.encoder_input = Input(shape=self.input_shape)
        self.encoder = LSTM(self.lstm_dim, return_state=True)
        # Only care about hidden states
        _, state_h, state_c = self.encoder(self.encoder_input)

        self.encoder_states = [state_h, state_c]

        # Init decoder
        self.decoder_input = Input(shape=self.input_shape)
        self.decoder = LSTM(self.lstm_dim, return_state=True,
                            return_sequences=True)
        # Only care about decoder output
        self.decoder_output, _, _ = self.decoder(
            self.decoder_input, initial_state=[state_h, state_c])
        # Decoder dense layer (final output layer)
        self.decoder_dense = Dense(self.output_size)
        model_output = self.decoder_dense(self.decoder_output)

        model = Model([self.encoder_input,
                       self.decoder_input], model_output)

        optimizer = keras.optimizers.SGD(
            lr=0.01, decay=0, momentum=0.8, nesterov=True)
        loss = tf.keras.losses.MeanSquaredError()

        model.compile(optimizer=optimizer, loss=loss)

        return model

    def init_inference_model(self):
        # Init encoder model
        encoder_model = Model(inputs=self.encoder_input,
                              outputs=self.encoder_states)
        tf.keras.utils.plot_model(
            encoder_model, to_file='encoder_model.png', show_shapes=True)

        # Init decoder model
        decoder_state_input_h = Input(shape=(self.lstm_dim,))
        decoder_state_input_c = Input(shape=(self.lstm_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_prediction, state_h, state_c = self.decoder(
            self.decoder_input, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_prediction)
        decoder_model = Model(
            [self.decoder_input] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def predict_seq(self, input_seq):
        encoder_model, decoder_model = self.init_inference_model()

        hidden_states = encoder_model.predict(input_seq)


scaler = sk.MinMaxScaler()
in_data = np.array([[1], [2], [3], [4], [5], [6]])
out_data = np.array([[2], [3], [4], [5], [6], [7]])

in_data = scaler.fit_transform(in_data)
out_data = scaler.fit_transform(out_data)

in_data = np.expand_dims(in_data, axis=0)
out_data = np.expand_dims(out_data, axis=0)
a = Seq2SeqModel(input_shape=(None, 1),
                 lstm_dim=100, output_size=1)
tf.keras.utils.plot_model(a.model, "test.png")
a.model.fit(x=[in_data, in_data], y=out_data, epochs=100)

print(a.model.summary())
