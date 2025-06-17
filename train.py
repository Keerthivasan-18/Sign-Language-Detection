import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Configurations
# -------------------------------
csv_path = 'asl_dataset.csv'  # Path to your CSV file
image_size = (64, 64)         # Resize images to 64x64
num_classes = 26              # A-Z

# -------------------------------
# Load CSV and Display Sample
# -------------------------------
data_df = pd.read_csv(csv_path)
print("Columns in DataFrame:", data_df.columns)

# Use the correct image path column (e.g., 'filename' or 'path')
image_col = 'filename'  # <- change this if your column is named differently

# Check if column exists
if image_col not in data_df.columns:
    raise ValueError(f"'{image_col}' column not found in CSV!")

# Show a random image
random_index = random.randint(0, len(data_df) - 1)
sample_path = data_df[image_col].iloc[random_index]
print("Showing image:", sample_path)

img = mpimg.imread(sample_path)
plt.imshow(img)
plt.axis('off')
plt.title(f"Label: {data_df['label'].iloc[random_index]}")
plt.show()

# -------------------------------
# Load and Preprocess Images
# -------------------------------
X = []
y = []

for i, row in data_df.iterrows():
    img_path = row[image_col]
    label = row['label']

    try:
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        X.append(img_array)
        y.append(ord(label.upper()) - ord('A'))  # Convert A-Z to 0-25
    except Exception as e:
        print(f"Skipping {img_path}: {e}")

X = np.array(X)
y = to_categorical(y, num_classes=num_classes)

print("Dataset loaded. X shape:", X.shape, "| y shape:", y.shape)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -------------------------------
# CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test), batch_size=32)

# -------------------------------
# Save Model
# -------------------------------
model.save('asl_model.h5')
print("Model saved as 'asl_model.h5'")
