# Required libraries
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import layers, Model, utils
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Load CSVs
df_images = pd.read_csv('data/clean/image_dataset.csv')
tabular_df = pd.read_csv('data/clean/tabular_data.csv')
df_images['filepath'] = df_images["filepath"].str.replace('/', "\\", regex=False)

# Merge on ID
merged_df = df_images.merge(tabular_df, on='id', how='inner')

# Keep numeric and boolean features
numeric_cols = [
    "track_number", "duration_ms", "popularity", "danceability", "energy", "loudness", 
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
    "time_signature", "is_explicit"
]
X_tab = merged_df[numeric_cols].values
y = merged_df["release_year"].values

y_min, y_max = np.min(y), np.max(y)
y_norm = (y - y_min) / (y_max - y_min)

# Scale tabular input
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)

# Load and resize images
def load_image(filepath):
    img = Image.open(filepath).convert("RGB").resize((64, 64))
    return np.array(img) / 255.0

X_img = np.stack([load_image(fp) for fp in merged_df['filepath']])

# Train/test split
X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    X_img, X_tab_scaled, y_norm, test_size=0.2, random_state=42)

# CNN branch
img_input = layers.Input(shape=(64, 64, 3), name="img_input")
x = layers.Conv2D(32, (3, 3), activation="relu")(img_input)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Tabular branch
tab_input = layers.Input(shape=(X_tab.shape[1],), name="tab_input")
t = layers.Dense(64, activation="relu")(tab_input)
t = layers.Dense(32, activation="relu")(t)

# Merge
combined = layers.concatenate([x, t])
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dense(1, activation="sigmoid")(z) 

# Model
model = Model(inputs=[img_input, tab_input], outputs=z)

model.summary()
model.save("figures/model.keras")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train
history = model.fit(
    {"img_input": X_img_train, "tab_input": X_tab_train},
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)

# Evaluate on normalized scale
loss, mae = model.evaluate(
    {"img_input": X_img_test, "tab_input": X_tab_test},
    y_test
)
print(f"[Normalized] Test Loss: {loss:.4f}, MAE: {mae:.4f}")

y_pred_norm = model.predict({"img_input": X_img_test, "tab_input": X_tab_test})
y_pred_year = y_pred_norm * (y_max - y_min) + y_min
y_test_year = y_test * (y_max - y_min) + y_min


for pred, true in zip(y_pred_year[:5].flatten(), y_test_year[:5]):
    print(f"Predicted: {pred:.1f}, True: {true:.1f}")

r2 = r2_score(y_test_year, y_pred_year)
print(f"R² score: {r2:.4f}")
