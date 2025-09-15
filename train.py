# ------------------ Import Libraries ------------------
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# ------------------ Parameters ------------------
sz = 64                # image size
batch_size = 32
epochs = 15
train_dir = "/content/drive/MyDrive/DatasetSign/train/"  # your training folder
test_dir = "/content/drive/MyDrive/DatasetSign/test/"    # your testing folder

# ------------------ Data Preprocessing ------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(sz, sz),
    batch_size=batch_size,
    color_mode='grayscale',      # assuming grayscale images
    class_mode='categorical'     # multi-class classification
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(sz, sz),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# ------------------ Build the CNN ------------------
model = Sequential()

# 1st conv layer
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(sz, sz, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd conv layer
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd conv layer (optional, deeper network)
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer (number of units = number of classes)
model.add(Dense(training_set.num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------ Train the CNN ------------------
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=epochs,
    validation_data=test_set,
    validation_steps=len(test_set)
)

# ------------------ Evaluate on test set ------------------
test_loss, test_acc = model.evaluate(test_set)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ------------------ Save the model ------------------
model.save("cnn_modelT.h5")
print("Model saved as cnn_model99.h5")

model.save_weights("cnn_weights99.h5")
