import pandas as pd
import numpy as np
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ✅ Load dataset
DATA_PATH = "all_data.csv"
df = pd.read_csv(DATA_PATH)

# ✅ Select only bands and spectral indices as features
BANDS_INDICES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDVI", "FDI", "PI", "NDWI", "WRI", "MNDWI", "SR", "RNDVI", "ARI",
    "MARI", "CHL_RedEdge", "REPI", "EVI", "EVI2", "GNDVI", "MCARI",
    "MSI", "NDMI", "NBR", "NDSI", "SAVI", "OSI", "PNDVI"
]

X = df[BANDS_INDICES]  # ✅ Keep only required features
y = df["label"]  # ✅ Labels

# ✅ Encode labels if necessary
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Initialize and train TabNet model
model = TabNetClassifier()
model.fit(
    X_train.values, y_train,
    max_epochs=10, patience=10, batch_size=256
)

# ✅ Save model using pickle
MODEL_PATH = "tabnet_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# ✅ Save label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ✅ Evaluate model
y_pred = model.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete. Accuracy: {accuracy:.4f}")
print(f"✅ Model saved as {MODEL_PATH}")
