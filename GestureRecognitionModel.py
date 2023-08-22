import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2

# Defining the path to the dataset
data_dir = '/Users/bretlotti/Documents/handGesturesTracking/HandGesture/images'
gesture_classes = sorted(os.listdir(data_dir))
num_classes = len(gesture_classes)
input_shape = (195, 240, 3)

# Loading and preprocessing the dataset
def load_and_preprocess_dataset(data_dir, input_shape):
    images = []
    labels = []
    for idx, gesture_class in enumerate(gesture_classes):
        class_dir = os.path.join(data_dir, gesture_class)
        if not os.path.isdir(class_dir):
            continue  # Skip non-directory entries
        for image_file in os.listdir(class_dir):
            if image_file == '.DS_Store':  # Skip .DS_Store files
                continue
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            image = cv2.resize(image, input_shape[:2])
            image = image / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(idx)
    return np.array(images), np.array(labels)

images, labels = load_and_preprocess_dataset(data_dir, input_shape)

# Splitting the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Building the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.summary()

# Define a function for data augmentation using TensorFlow operations
def data_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))  # Random rotation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

# Create a custom data pipeline using tf.data for data augmentation
def create_data_pipeline(X, y, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.map(lambda x, y: (data_augmentation(x, y)), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create data pipelines for training, validation, and test sets
batch_size = 32
train_data = create_data_pipeline(X_train, y_train, batch_size, augment=True)
val_data = create_data_pipeline(X_val, y_val, batch_size, augment=False)
test_data = create_data_pipeline(X_test, y_test, batch_size, augment=False)

# Train the model
epochs = 10

history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('gesture_recognition_model.h5')
