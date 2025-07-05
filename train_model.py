# import os
# import numpy as np
# import joblib
# from extract_features import extract_mfcc
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report


# def load_data(folder_path, label):
#     features = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".wav"):
#             filepath = os.path.join(folder_path, filename)
#             mfcc = extract_mfcc(filepath)
#             features.append(mfcc)
#     labels = [label] * len(features)
#     # print(features, labels)
#     return features, labels

# if __name__ == "__main__":
#     # Load user and impostor data
#     user_features, user_labels = load_data("user_voice", 1)
#     imp_features, imp_labels = load_data("impostor_voice", 0)

#     X = np.array(user_features + imp_features)
#     y = np.array(user_labels + imp_labels)

#     # Train KNN
#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(X, y)
#     preds = model.predict(X)
#     print("\n=== Training Report ===")
#     print(classification_report(y, preds))

#     # Save the model
#     joblib.dump(model, "voice_model.pkl")
#     print("✅ Model trained and saved as voice_model.pkl")



import os
import numpy as np
import joblib
from extract_features import extract_mfcc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_data(folder_path, label):
    features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            mfcc = extract_mfcc(filepath)
            features.append(mfcc)
    labels = [label] * len(features)
    return features, labels

# Load data
user_features, user_labels = load_data("user_voice", 1)
imp_features, imp_labels = load_data("impostor_voice", 0)

X = np.array(user_features + imp_features)
y = np.array(user_labels + imp_labels)

# Split into train/test (optional for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose model (uncomment one of the following):

# 1. Logistic Regression
# model = LogisticRegression()

# 2. Support Vector Machine (SVM)
model = SVC(kernel='linear', probability=True)

# 3. Random Forest
# model = RandomForestClassifier(n_estimators=50)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n=== Evaluation ===")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "voice_model.pkl")
print("✅ Model saved as voice_model.pkl")
