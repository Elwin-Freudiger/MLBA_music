# Required libraries
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, utils
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Load CSVs
df_images = pd.read_csv('data/clean/image_dataset.csv')
tabular_df = pd.read_csv('data/clean/tabular_data.csv')

# Merge on ID
merged_df = df_images.merge(tabular_df, on='id', how='left')

# Keep numeric and boolean features
numeric_cols = [
    "track_number", "duration_ms", "popularity", "danceability", "energy", "loudness", 
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
    "time_signature"
]
bool_cols = ["is_explicit"]
merged_df[bool_cols] = merged_df[bool_cols].astype(int)

X_tab = merged_df[numeric_cols + bool_cols].values
y = merged_df["release_year"].values

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
    X_img, X_tab_scaled, y, test_size=0.2, random_state=42
)

# Define CNN for image input
img_input = layers.Input(shape=(64, 64, 3), name="img_input")
x = layers.Conv2D(32, (3, 3), activation="relu")(img_input)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Define dense network for tabular input
tab_input = layers.Input(shape=(X_tab.shape[1],), name="tab_input")
t = layers.Dense(64, activation="relu")(tab_input)
t = layers.Dense(32, activation="relu")(t)

# Merge both
combined = layers.concatenate([x, t])
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dense(1, activation="linear")(z)  # regression output

# Build and compile model
model = Model(inputs=[img_input, tab_input], outputs=z)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Save model architecture image
utils.plot_model(model, to_file="model_architecture.png", show_shapes=True, dpi=100)

# Fit model
history = model.fit(
    {"img_input": X_img_train, "tab_input": X_tab_train},
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

# Evaluate
loss, mae = model.evaluate(
    {"img_input": X_img_test, "tab_input": X_tab_test},
    y_test
)
print(f"Test Loss: {loss:.2f}, MAE: {mae:.2f}")