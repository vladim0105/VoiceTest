
import warnings
warnings.simplefilter(
    action='ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, LSTM, Dense
import numpy as np


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
        self.predict_seq(a)

    def init_training_model(self):
        # Init encoder
        self.encoder_input = Input(shape=self.input_shape)
        self.encoder = LSTM(self.lstm_dim, return_state=True)
        # Only care about hidden states
        _, state_h, state_c = self.encoder(self.encoder_input)

        self.encoder_states = [state_h, state_c]

        # Init decoder
        self.decoder_input = Input(shape=self.input_shape)
        self.decoder = LSTM(self.lstm_dim, return_state=True, return_sequences=True)
        # Only care about decoder output
        self.decoder_output, _, _ = self.decoder(self.decoder_input, initial_state=[state_h, state_c])
        # Decoder dense layer (final output layer)
        self.decoder_dense = Dense(self.output_size)
        model_output = self.decoder_dense(self.decoder_output)

        model = Model(inputs=[self.encoder_input, self.decoder_input], outputs=model_output)

        optimizer = keras.optimizers.SGD(lr=0.1, decay=0, momentum=0.8, nesterov=True)
        loss = tf.keras.losses.MeanAbsoluteError()

        model.compile(optimizer=optimizer, loss=loss)

        return model

    def init_inference_model(self):
        # Init encoder model
        encoder_model = Model(inputs=self.encoder_input, outputs=self.encoder_states)
        tf.keras.utils.plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)

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


a = Seq2SeqModel(input_shape=(None, 257 * 2), lstm_dim=100, output_size=257 * 2)
