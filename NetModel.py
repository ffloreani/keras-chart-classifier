from keras import Sequential
from keras.layers import BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Conv2D, Dropout


# Original input size 750 x 500 (scaled to 224 x 224, convert RGB colors to single greyscale color)
def alex_net():
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, input_shape=(224, 224, 1), kernel_size=(11, 11), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(11, 11), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(units=1000))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.4))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(units=2, activation='softmax'))  # Chimeric, non-chimeric

    return model
