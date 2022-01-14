import tensorflow as tf
import numpy as np
import os
import pathlib
import skimage.io
import matplotlib.pyplot as plt
import PIL
import datetime
import PIL.Image
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

def parse_image(img_path: str) -> dict:
    train = True
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    mask_path = tf.strings.regex_replace(img_path, "img", "lbl")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return {'image': image, 'segmentation_mask': mask}



@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:

    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# def display_sample(display_list, dataset):
#     """Show side-by-side an input image,
#     the ground truth and the prediction.
#     """
#     plt.figure(figsize=(18, 18))

#     title = ['Input Image', 'True Mask', 'Predicted Mask']

#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         plt.axis('off')
#     plt.show()
# for image, mask in dataset['train'].take(1):
#     sample_image, sample_mask = image, mask   



# class DisplayCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         clear_output(wait=True)
#         show_predictions()
#         print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
# EPOCHS = 5

# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# callbacks = [
#     # to show samples after each epoch
#     DisplayCallback(),
#     # to collect some useful metrics and visualize them in tensorboard
#     tensorboard_callback,
#     # if no accuracy improvements we can stop the training directly
#     tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
#     # to save checkpoints
#     tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
# ]


# def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
#     """Return a filter mask with the top 1 predictions
#     only.
#     Parameters
#     ----------
#     pred_mask : tf.Tensor
#         A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
#         N_CLASS values (vector) which represents the probability of the pixel
#         being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
#         has been predicted class 2 with a probability of 100%.
#     Returns
#     -------
#     tf.Tensor
#         A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
#         for each pixels.
#     """
#     # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
#     # 1 prediction for each class but we want the highest score only
#     # so we use argmax
#     pred_mask = tf.argmax(pred_mask, axis=-1)
#     # pred_mask becomes [IMG_SIZE, IMG_SIZE]
#     # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
#     pred_mask = tf.expand_dims(pred_mask, axis=-1)
#     return pred_mask

# def show_predictions(dataset=None, num=1):
#     """Show a sample prediction.
#     Parameters
#     ----------
#     dataset : [type], optional
#         [Input dataset, by default None
#     num : int, optional
#         Number of sample to show, by default 1
#     """
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = model.predict(image)
#             display_sample([image[0], true_mask, create_mask(pred_mask)])
#     else:
#         # The model is expecting a tensor of the size
#         # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
#         # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
#         # and we want only 1 inference to be faster
#         # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
#         one_img_batch = sample_image[0][tf.newaxis, ...]
#         # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
#         inference = model.predict(one_img_batch)
#         # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
#         pred_mask = create_mask(inference)
#         # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
#         display_sample([sample_image[0], sample_mask[0],
#                         pred_mask[0]])
# for image, mask in dataset['train'].take(1):
#     sample_image, sample_mask = image, mask

# show_predictions()