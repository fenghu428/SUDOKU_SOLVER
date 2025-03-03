import cv2
import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import BatchNormalization, Dropout
from keras.models import Sequential

# convert image to greyscale and perform thresholding
def convert_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 125, 255,
                                cv2.THRESH_BINARY_INV)
    return np.array(thresh)

# load digits for model training
def load_digit_data():
    digits_directory = r'Digits/'
    categories = [str(number) for number in range(10)]
    digit_data = []

    for category in categories:
        folder_path = os.path.join(digits_directory, category)
        for image_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_filename) 
            label = int(category)  # labels are strings, convert to int
            image_array = cv2.imread(image_path)  
            resized_image = cv2.resize(image_array, (40, 40))  
            preprocessed_image = convert_image(resized_image)  
            digit_data.append([preprocessed_image, label])

    return digit_data

X = []
y = []
data = load_digit_data()

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# preprocess data
X = X.reshape(X.shape[0], -1)
X = X / 255
X = X.reshape(X.shape[0], 40, 40, 1)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# convert labels to one-hot encoding
train_y_one_hot = keras.utils.to_categorical(y_train)
test_y_one_hot = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(40,40,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, train_y_one_hot, epochs=10, batch_size=32, validation_data=(X_test, test_y_one_hot), verbose=1)

# Save the model
model.save('model.h5')