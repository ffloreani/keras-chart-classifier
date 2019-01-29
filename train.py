import os

from keras.optimizers import RMSprop

import DataGenerator
import Visualizer
from DataSplitter import split_data
from NetModel import alex_net

if __name__ == "__main__":

    model = alex_net()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    model.summary()

    # Split original_data into test & training sets
    print("[INFO] Splitting data to test & training sets...")
    split_data()

    # Create data batch generators
    print("[INFO] Creating data generators...")
    train_generator = DataGenerator.create_generator('./train_data')
    test_generator = DataGenerator.create_generator('./test_data')

    # Train model
    print('[INFO] Training model...')
    history = model.fit_generator(
        train_generator,
        epochs=50,
        verbose=True,
        shuffle=True,
        steps_per_epoch=32)

    # Evaluate model
    print('[INFO] Evaluating trained model...')
    (loss, accuracy) = model.evaluate_generator(
        generator=test_generator,
        steps=len(test_generator),
        verbose=True)

    print('Accuracy: {:.2f}%'.format(accuracy * 100))

    # Visualize training history
    Visualizer.draw_training_curve(history)

    # Save weights to file
    print('[INFO] Saving the model weights to file...')
    fileName = "./weights/alex_weights"
    if not os.path.exists(os.path.dirname(fileName)):
        os.path.makedirs(os.path.dirname(fileName))
    if os.path.isfile(fileName):
        os.remove(fileName)
    model.save_weights(fileName, overwrite=True)
