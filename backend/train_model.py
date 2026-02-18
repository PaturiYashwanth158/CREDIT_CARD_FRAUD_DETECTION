import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset (same folder)
data_path = os.path.join(BASE_DIR, "Credit_Card_Dataset.csv")
df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]

# Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        use_label_encoder=False
    ))
])

pipeline.fit(X_train, y_train)

print(classification_report(y_test, pipeline.predict(X_test)))

# Save model
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "ensemble_model.pkl")
joblib.dump(pipeline, model_path)

print("Model trained and saved successfully!")
