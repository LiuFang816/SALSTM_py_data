BASE_DIR = ".\\"
BASE_PREPROCESSEDIMAGES_DIR = BASE_DIR + "data_preprocessed_images\\"
BASE_TRAIN_SEGMENT_DIR = BASE_DIR + "data_segmenter_trainset\\"
PATIENT_PRED_DIR = BASE_DIR + "data_patient_predictions\\"

# Quick mode does away with training in different folds.
# It does overfit a little in the calibration and submission step.
# However it still scores ~0.010552 on the private LB which is enough for the 3rd place
# The advantages is that it takes only 4-5 hours to train and 1 hour to predict.
QUICK_MODE = True

MODEL_NAME = "model_quick" if QUICK_MODE else "model_full"
TRAIN_EPOCHS = 20 if QUICK_MODE else 30
FOLD_COUNT = 6
TARGET_SIZE = 256
CROP_INDENT_X = 32
TARGET_CROP = 184
CROP_INDENT_Y = 32 - ((TARGET_CROP - 160) / 2)
CROP_SIZE = 16

