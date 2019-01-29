from keras_preprocessing.image import ImageDataGenerator


def create_generator(image_dir):
    data_gen = ImageDataGenerator(
        horizontal_flip=True,  # Flip images to create more original_data
        rescale=1. / 255  # Scale down values to [0,1] range for better handling
    )

    return data_gen.flow_from_directory(
        directory=image_dir,
        target_size=(224, 224),
        color_mode='grayscale',
        class_mode='categorical')
