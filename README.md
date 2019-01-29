# Keras overlap chart classifier

Deep learning image classifier written in Keras, used for detection of chimeric signals in genome overlap charts.

Created as a part of the master thesis project at the Faculty of Electrical Engineering & Computing, University of Zagreb.
Many thanks to Lovro Vrček, mag. phys. and prof. dr. sc. Mile Šikić for their help & guidance.

## Data preprocessing

As the original data is stored in 3 separate folders, sorted by their overlap type (see `original_data` folder), the first step is to split it into separate training & test data sets. As the idea of the whole network is to find out only if an image represents a chimeric overlap or not, the regular & repeat overlaps are stored in an adjoining folder (labeled `non_chimeric`), with the chimeric overlaps remaining separate (labeled `chimeric`). The training/test split percentage is set to 75/25 (75% of data for training, 25% for testing). 

It should be noted that the data is randomly re-sorted with each new training run.

### Data generators

The sorted data is fed to the model via the Keras ImageDataGenerator class ([docs](https://keras.io/preprocessing/image/#imagedatagenerator-class)). Besides feeding the data to the model in training, the generators have an additional task of modifying the data by invoking a series of transformations:

1. Horizontal flip (Generating double the amount of original data by mirroring the image along it's Y axis)
2. Rescaling (The original images are 8-bit RGB encoded, resulting in too much data, so the color channels are scaled to the [0-1] range)
3. Scaling (Shrinking the images from the original 750 x 500 px to 224 x 224 px)
4. Recoloring (Merging the RGB color channels into a single greyscale channel)

The resulting images are now ready to be loaded into the model.

## Model summary

The learning model is based on the AlexNet image classification network ([Original author's presentation from ImageNet](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf)), although it has been a bit trimmed to bring down the number of training parameters (currently at ~63.000.000 trainable params).

If you're looking for a more verbose summary, you can have a look at the model output in the image below:

![Model summary](https://i.imgur.com/uQm4ov6.png)

## Training & evaluation

Training is done by fitting the model to the data received from the image generators. The generators create batches of 32 images per epoch. To match the generated batches, there are 32 steps in every epoch of the fitting. 
The collected metrics are visualized on a plot at the end of the evaluation.

* **Loss function** = Categorical cross-entropy
* **Optimizer** = RMS prop with a learning rate of e^-4
* **Metrics** = Loss value, accuracy

### Results after 5 epochs with 50 steps each (5 epochs only due to lack of testing hardware) 
![Accuracy](https://i.imgur.com/B5qrmoe.png)

### Hardware configuration

Apple Macbook Pro (2017):
 * Intel I7-7567U
 * 16 GB RAM
 * No dedicated GPU

Training time rounds up to ~ 27s per single epoch step (5 epochs add up to about 2h runtime).
