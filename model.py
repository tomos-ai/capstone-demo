import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as kbackend
import numpy as np
import cv2
from util import np_to_base64

# Force CPU usage
# https://stackoverflow.com/a/70015395/1071459
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# IMPORTANT!!
#   All models are loaded from the same model, weights will be loaded in a next step
model_dict_by_name = {
    'v2_frozen_reduced_5percent_model_resnet50_extended_functional_imgsize256x256_best':
    tf.keras.models.load_model(
        'models/v2_frozen_reduced_5percent_model_resnet50_extended_functional_imgsize256x256_best.hdf5'
    ),
    'semi_sup_best':
    tf.keras.models.load_model(
        'models/v2_frozen_reduced_5percent_model_resnet50_extended_functional_imgsize256x256_best.hdf5'
    ),
    'covid-oct_transfer':
    tf.keras.models.load_model(
        'models/v2_frozen_reduced_5percent_model_resnet50_extended_functional_imgsize256x256_best.hdf5'
    ),
}
# Setting weights for other models
model_dict_by_name['covid-oct_transfer'].load_weights(
    'models/weights_covid-oct_transfer.h5')
model_dict_by_name['semi_sup_best'].load_weights(
    'models/weights_semi_sup_best.h5')


def model_predict_with_heatmap(img, model_name):
    model = model_dict_by_name[model_name] # Model to use

    img = img.resize((256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)

    if x.shape[2] == 3: # Color image
        x = np.mean(x, axis=2)
    elif x.shape[2] == 4: # Color image with alpha channel
        x = np.mean(x[:, :, 0:3], axis=2)

    x = np.array([x, x, x]) # Our model uses the same value for rgb
    x = np.squeeze(x) # Remove extra dimension
    x = x.transpose([1, 2, 0]) # Since x ([x,x,x]), we need to reorder the channels to (256,256,3)
    x = np.expand_dims(x, axis=0) # Needed for batch processing
    x_unprocessed = np.copy(x) # Needed for the heatmap

    x = tf.keras.applications.resnet.preprocess_input(x)

    preds = model.predict(x)

    class_dict = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3} # From training
    class_dict_rev = {num: c for c, num in class_dict.items()}

    predicted_position = preds.argmax()

    img_batch, img_batch_with_heatmap, heatmaps = get_imgs_and_heatmaps(
        model,
        'conv5_block3_out',
        x_unprocessed,
        intensity=0.5,
        preprocess_function=tf.keras.applications.resnet.preprocess_input,
    )

    return (class_dict_rev[predicted_position], preds[0][predicted_position],
            img_batch_with_heatmap[0])


def get_heatmaps(functional_model,
                 last_conv_layer_name,
                 img_batch,
                 preprocess_function=None):
    """
    Matrix implementation of heatmap using Grad-CAM. Obtains heatmaps for the given images batches.

    Based on https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759
    """
    img_batch_copy = np.copy(img_batch)
    if preprocess_function is not None:
        img_batch_copy = preprocess_function(img_batch_copy)

    with tf.GradientTape() as tape:
        last_conv_layer = functional_model.get_layer(last_conv_layer_name)
        iterate = tf.keras.models.Model(
            [functional_model.inputs],
            [functional_model.output, last_conv_layer.output])
        model_out, last_conv_layer_out = iterate(img_batch_copy)
        max_pos_arr = np.argmax(model_out, axis=-1)
        class_out = [
            model_out[index, max_pos]
            for index, max_pos in enumerate(max_pos_arr)
        ]
        grads = tape.gradient(class_out, last_conv_layer_out)
        pooled_grads = kbackend.mean(grads, axis=(1, 2))

    pooled_grads_expanded = tf.expand_dims(pooled_grads, axis=1)  # Expand once
    pooled_grads_expanded = tf.expand_dims(pooled_grads_expanded,
                                           axis=1)  # Expand twice
    heatmaps = tf.reduce_mean(tf.multiply(pooled_grads_expanded,
                                          last_conv_layer_out),
                              axis=-1)
    heatmaps = np.maximum(heatmaps, 0)
    heatmaps_max_expanded = tf.expand_dims(np.max(heatmaps, axis=(1, 2)),
                                           axis=-1)  # Expand once
    heatmaps_max_expanded = tf.expand_dims(heatmaps_max_expanded,
                                           axis=-1)  # Expand twice
    heatmaps = heatmaps / heatmaps_max_expanded
    return heatmaps.numpy()


def get_imgs_and_heatmaps(
    functional_model,
    last_conv_layer_name,
    img_batch,
    intensity=0.5,
    preprocess_function=None,
):
    """
    Returns images with the Grad-CAM heatmaps on top.

    Returns img_batch, img_batch_with_heatmap, heatmaps
    """
    heatmaps = get_heatmaps(
        functional_model,
        last_conv_layer_name,
        img_batch=img_batch,
        preprocess_function=preprocess_function,
    )

    img_batch_copy = np.copy(img_batch)
    img_batch_with_heatmap = np.zeros_like(img_batch_copy)
    for index, img in enumerate(img_batch_copy):
        heatmap_img = cv2.resize(heatmaps[index], (img.shape[1], img.shape[0]))
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_img),
                                        cv2.COLORMAP_JET)
        img_with_heatmap = heatmap_img * intensity + img
        img_bgr = np.clip(img_with_heatmap, a_min=0, a_max=255)
        img_rgb = img_bgr[..., ::-1]
        img_batch_with_heatmap[index] = np.int0(img_rgb)

    return img_batch_copy, img_batch_with_heatmap, heatmaps
