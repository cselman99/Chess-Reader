import sys
import numpy as np
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import cv2
import os
import tensorflow as tf
from absl import logging
import workspace.Constants as Constants
from workspace.Computer_Vision.ChessDriver import predictSquare
# from workspace.Computer_Vision.ChessboardDetection import getHoughLines

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

PATH_TO_DIR = r'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/imgs'
train_path = f"{PATH_TO_DIR}/train/"
test_path = f"{PATH_TO_DIR}/test/"

model_path = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/models/det3.tflite'  # Rename to model name in this folder
DETECTION_THRESHOLD = 0.5
EPOCHS = 40
BATCH_SIZE = 2

# Pick between efficientdet_lite0, efficientdet_lite1, efficientdet_lite2, efficientdet_lite3
MODEL = 'efficientdet_lite3'

def _run():
    """
    Load training data from path and train specified model. Exports model when finished.
    :return: None
    """

    os.environ["TFHUB_CACHE_DIR"] = 'C:/Users/Carter/Desktop/Classes/Chess-Reader/workspace/Object_Detection/temp_model_storage'
    spec = model_spec.get(MODEL)

    print(f'Building {MODEL.upper()} with BATCH SIZE: {BATCH_SIZE} | EPOCHS: {EPOCHS} | DETECTION THRESHOLD: {DETECTION_THRESHOLD}')

    test_data = object_detector.DataLoader.from_pascal_voc(images_dir=test_path + 'img',
                                                          annotations_dir=test_path + 'anno',
                                                          label_map=Constants.label_map)

    train_data = object_detector.DataLoader.from_pascal_voc(images_dir=train_path + 'img',
                                                            annotations_dir=train_path + 'anno',
                                                            label_map=Constants.label_map)

    model = object_detector.create(train_data, model_spec=spec, batch_size=BATCH_SIZE, train_whole_model=True, epochs=EPOCHS)
    # metrics = model.evaluate(test_data)
    metrics = model.evaluate_tflite(model_path, test_data)
    print(metrics)

    # model.export(export_dir='.')


def _preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def _detect_objects(interpreter, image, threshold):
    """
    Returns a list of detection results, each a dictionary of object info.
    """

    signature_fn = interpreter.get_signature_runner()
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i]
            }
            results.append(result)
    return results


def _run_odt(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    # Load the labels into a list
    classes = [''] * 13
    for label_id, label_name in Constants.label_map.items():
        classes[label_id - 1] = label_name

    # Load the input image and preprocess it
    preprocessed_image, original_image = _preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)

    # Run object detection on the input image
    results = _detect_objects(interpreter, preprocessed_image, threshold=threshold)
    return original_image_np, results


def detect(fp, interpreter):
    """Run the Object-Detection model on Image and Location analysis"""
    pieces = []

    original_image_np, results = _run_odt(fp, interpreter, threshold=DETECTION_THRESHOLD)
    # ------------------------------------------------ #
    # Gather Hough-Lines from image
    # img = cv2.imread(fp)
    # houghlines = getHoughLines(img)

    # Confirm correct number of hough lines
    # if len(houghlines[0]) != 8 or len(houghlines[1] != 8):
    #     print("Wrong number of hough lines detected")
    #     # return pieces
    # houghlines[0] = houghlines[0][:9]
    # houghlines[1] = houghlines[1][:9]
    # Get Piece Location
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        classID = obj['class_id']
        className = Constants.label_map[int(classID) + 1]

        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])
        width = xmax - xmin
        height = ymax - ymin

        centroid = (int(xmin + (width / 2)), int(ymin + (height / 1.5)))
        print(className, centroid)
        try:
            square = predictSquare(centroid)
            pieces.append((className, square))
        except Exception as e:
            print("failed to predict square for " + className)
            print(e)

    return pieces


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '0':
        _run()
    # ------------------------------------------------ #
    # Load the TFLite model
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    # # ------------------------------------------------ #
    # fp = f"{test_path}img/fdcd6ada676799da8a870f58fdf548db_jpg.rf.54abced68347da874d25c5d3886d3c4a.jpg"
    # detect(fp, interpreter)

