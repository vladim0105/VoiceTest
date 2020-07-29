
import warnings
warnings.simplefilter(
    action='ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, LSTM, TimeDistributed, Dense
import librosa
import preprocess
import numpy as np
import sklearn.preprocessing as sk
import glob
import sys
from sklearn.metrics import mean_squared_error


def stft_to_complex(stft):
    """Converts the [real, imag] values in the stft to complex numbers"""
    complex_stft = np.zeros(
        (stft.shape[0], stft.shape[1]), dtype=np.complex64)  # pylint: disable=E1136

    for i in range(stft.shape[0]):
        for j in range(stft.shape[1]):
            real_stft = stft[i, j]
            complex_stft[i, j] = complex(
                real_stft[0], real_stft[1])
    return complex_stft


def decoded_to_wav(decoded_data, file_name):
    stft = decoded_data.reshape((decoded_data.shape[0], int(514 / 2), 2))
    transposed_stft = np.transpose(stft, (1, 0, 2))
    complex_stft = stft_to_complex(transposed_stft)
    wave_data = librosa.istft(complex_stft, win_length=256)
    librosa.output.write_wav(file_name, wave_data, sr=22050)


test_data = np.load(
    "data/stft_loss/sample-000000.npy")


file_names = [f.split("\\")[1].split(".")[0]
              for f in glob.glob("data/stft_loss/*.npy")]

decoded_to_wav(test_data, "test.wav")
epochs = 1
batch_size = 2
timesteps = 250
MAX_TIMESTEPS = 5000
lstm_dim = 400
assert MAX_TIMESTEPS % (batch_size * timesteps) == 0, "Make sure that timesteps and batch size can fit fully inside a sample."


def scale_data(data):
    x_scaler = sk.StandardScaler()
    y_scaler = sk.StandardScaler()

    x_scaler = x_scaler.fit(data[:, :])
    y_scaler = y_scaler.fit(data[:, :])

    scaled_data = x_scaler.transform(data)
    return scaled_data


def data_generator(sample_names, epochs, batch_size, timesteps):
    for n_epoch in range(epochs):
        n_batch = 0
        original_batch = np.zeros((batch_size, timesteps, 257 * 2))
        loss_batch = np.zeros((batch_size, timesteps, 257 * 2))
        target_batch = np.zeros((batch_size, timesteps, 257 * 2))
        for sample_idx in range(len(sample_names)):
            sample_name = sample_names[sample_idx]
            original_data = scale_data(np.load("data/stft_original/" + sample_name + ".npy"))
            loss_data = scale_data(np.load("data/stft_loss/" + sample_name + ".npy"))
            # One time-step ahead
            target_data = np.pad(original_data[1:], ((0, 1), (0, 0)))
            assert original_data.shape == loss_data.shape == target_data.shape == test_data.shape
            for n_timesteps in range(0, MAX_TIMESTEPS, timesteps):
                for n_timestep in range(0, timesteps):
                    original_batch[n_batch, n_timestep] = original_data[n_timesteps + n_timestep]
                    loss_batch[n_batch, n_timestep] = loss_data[n_timesteps + n_timestep]
                    target_batch[n_batch, n_timestep] = target_data[n_timesteps + n_timestep]
                n_batch += 1
                if n_batch == batch_size:
                    n_batch = 0
                    yield ([loss_batch, original_batch], target_batch)


def build_model():
    encoder_input = Input(shape=(None, 257 * 2))
    encoder = LSTM(lstm_dim, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None, 257 * 2))
    decoder = LSTM(lstm_dim, return_state=True, return_sequences=True)
    decoder_output, _, _ = decoder(decoder_input, initial_state=[state_h, state_c])
    decoder_dense = Dense(514)
    model_output = decoder_dense(decoder_output)
    model = Model([encoder_input, decoder_input], model_output)

    return model, encoder_states


lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", verbose=1, factor=0.5, patience=1)
optimizer = keras.optimizers.SGD(lr=0.1, decay=0, momentum=0.8, nesterov=True)
mse = tf.keras.losses.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse)


training_sample_names = file_names[:1]
validation_sample_names = file_names[:1]


# Inference


def decode(input_data):
    encoder_model = Model(encoder_input, encoder_states)

    decoder_state_input_h = Input(shape=(lstm_dim,))
    decoder_state_input_c = Input(shape=(lstm_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(
        decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    states_value = encoder_model.predict(np.expand_dims(input_data, axis=1))

    target_seq = np.zeros((1, 1, 257 * 2))
    target_seq[0, 0] = input_data[0]
    decoded_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        target_seq[0, 0] = input_data[i]
        frame_output, h, c = decoder_model.predict(
            [target_seq] + states_value)

        decoded_data[i] = frame_output

        states_value = [h, c]

    return decoded_data


training_steps = ((len(training_sample_names) * MAX_TIMESTEPS) / timesteps) / batch_size
model.fit(x=data_generator(training_sample_names, epochs, batch_size, timesteps), epochs=epochs, steps_per_epoch=training_steps,
          batch_size=batch_size, callbacks=[lr])
# TODO Scale, then unscale test data
scaler = sk.StandardScaler()
scaler = scaler.fit(test_data)
scaled = scaler.transform(test_data)
a = decode(scaled[:timesteps])
print(np.max(a))
inv = scaler.inverse_transform(a)
decoded_to_wav(inv, "result.wav")
print(inv[0, 0], inv[1, 1], inv[1, 2], inv[1, -1])
print(test_data[0, 0], test_data[0, 1], test_data[0, 2], test_data[0, -1])
mse = mean_squared_error(inv[:timesteps], test_data[:timesteps])
print(mse)
print(np.argmax(inv), np.argmax(test_data))
model.save("checkpoint.h5")
