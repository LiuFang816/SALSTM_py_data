import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam


def generate_saliceny_map():
    """Generates a heatmap indicating the pixels that contributed the most towards
    maximizing the filter output. First, the class prediction is determined, then we generate heatmap
    to visualize that class.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        seed_img = utils.load_img(path, target_size=(224, 224))
        pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
        heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
        cv2.imshow('Saliency - {}'.format(utils.get_imagenet_label(pred_class)), heatmap)
        cv2.waitKey(0)


def generate_cam():
    """Generates a heatmap via grad-CAM method.
    First, the class prediction is determined, then we generate heatmap to visualize that class.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Tigerwater_edit2.jpg/170px-Tigerwater_edit2.jpg']:
        seed_img = utils.load_img(path, target_size=(224, 224))
        pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
        cv2.imshow('Attention - {}'.format(utils.get_imagenet_label(pred_class)), heatmap)
        cv2.waitKey(0)


if __name__ == '__main__':
    generate_cam()
