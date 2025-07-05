import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_mfcc(filename, show_plot=False):
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Take average of coefficients
    if show_plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
    return mfcc_mean

if __name__ == "__main__":
    filename = "voice_sample.wav"
    features = extract_mfcc(filename, show_plot=True)
    print("Extracted MFCC features:", features)
