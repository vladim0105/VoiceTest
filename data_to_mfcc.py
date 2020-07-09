import librosa
import numpy as np
y, sr = librosa.load("data/raw_data/sample-000000.mp3")

stft = librosa.stft(y, n_fft=512, win_length=512)

real_stft = np.zeros(
    (stft.shape[0], stft.shape[1], 2))  # pylint: disable=E1136

for idx, x in np.ndenumerate(stft):
    real_stft[idx[0], idx[1], 0] = x.real
    real_stft[idx[0], idx[1], 1] = x.imag

print(real_stft)
np.save("test", real_stft)
istft = librosa.istft(stft)
librosa.output.write_wav("test.wav", istft, sr)
