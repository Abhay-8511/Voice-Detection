import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = joblib.load("training_data.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

joblib.dump(model, "voice_detector.pkl")

print(f"✅ Model trained successfully")
print(f"🎯 Validation Accuracy: {acc:.2f}")
