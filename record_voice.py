import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(filename='papa.wav', duration=3, fs=44100):
    print("Recording... Speak now.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)
    print(f"Saved recording as {filename}")

if __name__ == "__main__":
    record_voice()
