"""
File to predict the decade instead of the year. This becomes a classification task.
"""

#import libraries
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

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
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)

merged_df["decade"] = (merged_df["release_year"] // 10) * 10


# Downsample each group to min_count
min_count = merged_df["decade"].value_counts().min()

balanced_df = (
    merged_df
    .groupby("decade", group_keys=False)
    .apply(lambda x: x.sample(min_count, random_state=42))
    .reset_index(drop=True)
)

decade_labels, decade_classes = pd.factorize(merged_df["decade"])
y = to_categorical(decade_labels)

def load_image(filepath):
    img = Image.open(filepath).convert("RGB").resize((64, 64))
    return np.array(img) / 255.0

X_img = np.stack([load_image(fp) for fp in merged_df['filepath']])

# Split
X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    X_img, X_tab_scaled, y, test_size=0.2, random_state=42
)

# CNN input
img_input = layers.Input(shape=(64, 64, 3), name="img_input")
x = layers.Conv2D(32, (3, 3), activation="relu")(img_input)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Tabular input
tab_input = layers.Input(shape=(X_tab.shape[1],), name="tab_input")
t = layers.Dense(64, activation="relu")(tab_input)
t = layers.Dense(32, activation="relu")(t)

# Combine
combined = layers.concatenate([x, t])
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dense(y.shape[1], activation="softmax")(z)  # multi-class

# Compile
model = Model(inputs=[img_input, tab_input], outputs=z)

#add model summary
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    {"img_input": X_img_train, "tab_input": X_tab_train},
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)

# Evaluate
loss, accuracy = model.evaluate(
    {"img_input": X_img_test, "tab_input": X_tab_test},
    y_test
)

# Predict decades
y_pred = model.predict({"img_input": X_img_test, "tab_input": X_tab_test})
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

y_pred_decades = decade_classes[y_pred_labels]
y_true_decades = decade_classes[y_true_labels]

# Show 5 predictions
for pred, true in zip(y_pred_decades[:5], y_true_decades[:5]):
    print(f"Predicted: {pred}s, True: {true}s")

print(f"Final Test Accuracy: {accuracy:.3f}")
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=decade_classes, yticklabels=decade_classes
)
plt.title("Confusion Matrix (Decade Classification)")
plt.xlabel("Predicted Decade")
plt.ylabel("True Decade")
plt.tight_layout()
plt.show()
