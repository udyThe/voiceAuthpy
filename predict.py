import joblib
from extract_features import extract_mfcc

def predict_voice(filename):
    model = joblib.load("voice_model.pkl")
    features = extract_mfcc(filename).reshape(1, -1)
    # print("Extracted features:", features)
    prediction = model.predict(features)[0]
    if prediction == 1:
        print("✅ Access Granted: Voice matched.")
    else:
        print("❌ Access Denied: Voice not recognized.")

if __name__ == "__main__":
    test_file = "mummy.wav"
    predict_voice(test_file)
