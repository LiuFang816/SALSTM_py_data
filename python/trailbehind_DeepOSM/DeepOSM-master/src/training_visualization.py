"""Visualize predictions from a neural net analyzing satellite imagery to extract features."""

from __future__ import print_function
import numpy
import os
import time
from PIL import Image
from src.training_data import load_training_tiles, way_bitmap_for_naip
from src.single_layer_network import list_findings


def render_errors(raster_data_paths, model, training_info, render_results):
    """Render JPEGs showing findings."""
    for path in raster_data_paths:
        labels, images = load_training_tiles(path)
        if len(labels) == 0 or len(images) == 0:
            print("WARNING, there is a borked naip image file")
            continue
        false_positives, fp_images = list_findings(labels, images, model)
        path_parts = path.split('/')
        filename = path_parts[len(path_parts) - 1]
        print("FINDINGS: {} false pos of {} tiles, from {}".format(
            len(false_positives), len(images), filename))
        render_results_for_analysis([path], false_positives, fp_images, training_info['bands'],
                                    training_info['tile_size'])


def render_results_for_analysis(raster_data_paths, predictions, test_images, band_list, tile_size):
    """Generate a JPEG for each TIFF showing predictions shaded."""
    for raster_data_path in raster_data_paths:
        way_bitmap_npy = numpy.asarray(way_bitmap_for_naip(None, raster_data_path, None, None,
                                                           None))
        render_predictions(raster_data_path, predictions, test_images, way_bitmap_npy, band_list,
                           tile_size)


def render_predictions(raster_data_path, predictions, test_images, way_bitmap_npy, band_list,
                       tile_size):
    """Generate a JPEG for the given raster_data_path, showing predictions shaded."""
    test_images_by_naip = []
    predictions_by_naip = []

    index = 0
    for image_info in test_images:
        predictions_by_naip.append(predictions[index])
        test_images_by_naip.append(test_images[index])
        index += 1

    render_results_as_image(raster_data_path,
                            way_bitmap_npy,
                            test_images_by_naip,
                            band_list,
                            tile_size,
                            predictions=predictions_by_naip)


def render_results_as_image(raster_data_path,
                            way_bitmap,
                            test_images,
                            band_list,
                            tile_size,
                            predictions=None):
    """Save the source TIFF as a JPEG, with labels and data overlaid."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outfile = os.path.splitext(raster_data_path)[0] + '-' + timestr + ".jpeg"
    # TIF to JPEG bit from:
    # http://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
    im = Image.open(raster_data_path)
    print("GENERATING JPEG for %s" % raster_data_path)
    rows = len(way_bitmap)
    cols = len(way_bitmap[0])
    t0 = time.time()
    r, g, b, ir = im.split()
    # visualize single band analysis tinted for R-G-B,
    # or grayscale for infrared band
    if sum(band_list) == 1:
        if band_list[3] == 1:
            # visualize IR as grayscale
            im = Image.merge("RGB", (ir, ir, ir))
        else:
            # visualize single-color band analysis as a scale of that color
            zeros_band = Image.new('RGB', r.size).split()[0]
            if band_list[0] == 1:
                im = Image.merge("RGB", (r, zeros_band, zeros_band))
            elif band_list[1] == 1:
                im = Image.merge("RGB", (zeros_band, g, zeros_band))
            elif band_list[2] == 1:
                im = Image.merge("RGB", (zeros_band, zeros_band, b))
    else:
        # visualize multi-band analysis as RGB
        im = Image.merge("RGB", (r, g, b))

    t1 = time.time()
    print("{0:.1f}s to FLATTEN the {1} analyzed bands of TIF to JPEG".format(t1 - t0, sum(
        band_list)))

    t0 = time.time()
    shade_labels(im, test_images, predictions, tile_size)
    t1 = time.time()
    print("{0:.1f}s to SHADE PREDICTIONS on JPEG".format(t1 - t0))

    t0 = time.time()
    # show raw data that spawned the labels
    for row in range(0, rows):
        for col in range(0, cols):
            if way_bitmap[row][col] != 0:
                im.putpixel((col, row), (255, 0, 0))
    t1 = time.time()
    print("{0:.1f}s to DRAW WAYS ON JPEG".format(t1 - t0))

    im.save(outfile, "JPEG")


def shade_labels(image, labels, predictions, tile_size):
    """Visualize predicted ON labels as blue, OFF as green."""
    label_index = 0
    for label in labels:
        start_x = label[1][0]
        start_y = label[1][1]
        for x in range(start_x, start_x + tile_size):
            for y in range(start_y, start_y + tile_size):
                r, g, b = image.getpixel((x, y))
                if predictions[label_index][0] < predictions[label_index][1]:
                    # shade ON predictions blue
                    image.putpixel((x, y), (r, g, 255))
                else:
                    # shade OFF predictions green
                    image.putpixel((x, y), (r, 255, b))
        label_index += 1
