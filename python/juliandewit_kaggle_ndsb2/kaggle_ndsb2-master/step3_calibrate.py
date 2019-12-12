import helpers
import settings
import pandas
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

MODEL_NAME = settings.MODEL_NAME
SRC_PATH = settings.BASE_DIR + "prediction_raw_" + MODEL_NAME + ".csv"
PREDICT_RATIO = False

pred_data = pandas.read_csv(SRC_PATH, sep=";")
print str(len(pred_data)) + " rows"


fold_ranges = [
    ["fold0", 1, 140],
    ["fold1", 141, 280],
    ["fold2", 281, 420],
    ["fold3", 421, 560],
    ["fold4", 561, 700],
    ["full", 701, 1140]
]

ext = ""
for dia_sys in ["dia" + ext, "sys" + ext]:
    new_predictions = []
    real_value_col = "Diastole" if dia_sys.startswith("dia") else "Systole"
    mul_col = "mul_" + dia_sys
    pred_data[mul_col] = pred_data[real_value_col] / pred_data["pred_" + dia_sys]
    sex_map = {"M": 1, "F": 0}
    pred_data["sex_val"] = pred_data["sex"].map(sex_map)
    plane_map = {"ROW": 1, "COL": 0 }
    pred_data["plane_val"] = pred_data["plane"].map(plane_map)
    pred_data["pixels"] = pred_data["rows"] * pred_data["columns"] * pred_data["spacing"]
    print
    print "**** " + dia_sys + " ****"
    print

    print "\t".join(map(lambda x: str(x).rjust(10), ["Fold", "Train-n", "Validate-n", "Raw", "Raw^2", "Raw mi", "Raw mx", "Train", "Train^2", "Train mi", "Train mx", "Validate", "Validate^2", "Validate mi", "Validate mx", "MAE"]))
    for fold_range in fold_ranges:
        print_values = [fold_range[0]]
        feature_names = ["rows", "columns", "spacing", "slice_thickness", "slice_count", "up_down_agg", "age_years", "small_slice_count", "pred_sys" + ext, "pred_dia" + ext, "angle", real_value_col, mul_col]
        if PREDICT_RATIO:
            pred_data["error_" + dia_sys] = pred_data[real_value_col] * 100 / pred_data["pred_" + dia_sys]
        tmp_train = pred_data[((pred_data["patient_id"] < fold_range[1]) | (pred_data["patient_id"] > fold_range[2])) & (pred_data["patient_id"] <= 700) & (pred_data["slice_count"] > 7) ]  # patient 595 and 599 have invalid recordings

        if len(tmp_train[tmp_train["pred_dia"] == 0]) > 0:
            print "Warning '0' predictions"
            print tmp_train[tmp_train["pred_dia"] == 0]["patient_id"]

        if len(tmp_train[tmp_train["pred_sys"] == 0]) > 0:
            print "Warning '0' predictions"
            print tmp_train[tmp_train["pred_sys"] == 0]["patient_id"]

        x_train = tmp_train[feature_names]
        y_train = tmp_train["error_" + dia_sys]
        print_values.append(len(x_train))
        tmp_validate = pred_data[(pred_data["patient_id"] >= fold_range[1]) & (pred_data["patient_id"] <= fold_range[2]) ]

        x_validate = tmp_validate[feature_names]
        y_validate = tmp_validate["error_" + dia_sys]
        print_values.append(len(x_validate))

        print_values.append(round(y_validate.abs().mean(), 2))
        print_values.append(round(y_validate.apply(lambda xx: xx * xx).mean(), 2))
        print_values.append(round(y_validate.min(), 2))
        print_values.append(round(y_validate.max(), 2))

        del x_train[real_value_col]
        real_values = x_validate[real_value_col]
        del x_validate[real_value_col]
        del x_validate[mul_col]
        del x_train[mul_col]

        cls = GradientBoostingRegressor(learning_rate=0.001, n_estimators=2500, verbose=False, max_depth=3, min_samples_leaf=2, loss="ls", random_state=1301)
        cls.fit(x_train, y_train)
        y_pred = cls.predict(x_train)

        print_values.append(round(mean_absolute_error(y_train, y_pred), 2))
        print_values.append(round(mean_squared_error(y_train, y_pred), 2))
        print_values.append(round((y_train - y_pred).min(), 2))
        print_values.append(round((y_train - y_pred).max(), 2))

        y_pred = cls.predict(x_validate)

        print_values.append(round(mean_absolute_error(y_validate.fillna(0), y_pred), 2))
        print_values.append(round(mean_squared_error(y_validate.fillna(0), y_pred), 2))
        print_values.append(round((y_validate.fillna(0) - y_pred).min(), 2))
        print_values.append(round((y_validate.fillna(0) - y_pred).max(), 2))

        if PREDICT_RATIO:
            print_values.append(round(mean_absolute_error(x_validate["pred_" + dia_sys] * y_pred / 100, real_values.fillna(0)), 2))
            new_predictions += (x_validate["pred_" + dia_sys] * y_pred / 100).map(lambda x: round(x, 2)).values.tolist()
        else:
            print_values.append(round(mean_absolute_error(x_validate["pred_" + dia_sys] - y_pred, real_values.fillna(0)), 2))
            new_predictions += (x_validate["pred_" + dia_sys] - y_pred).map(lambda x: round(x, 2)).values.tolist()

        # print
        print "\t".join(map(lambda x: str(x).rjust(10), print_values))

    pred_data["cal_pred_" + dia_sys] = new_predictions
    pred_data["cal_error_" + dia_sys] = new_predictions - pred_data[real_value_col]
    pred_data["cal_abserr_" + dia_sys] = abs(pred_data["cal_error_" + dia_sys])


pred_data = pred_data[["patient_id", "slice_count", "age_years", "sex", "normal_slice_count", "Diastole", "Systole", "cal_pred_dia", "cal_error_dia", "cal_abserr_dia", "cal_pred_sys", "cal_error_sys", "cal_abserr_sys" , "pred_dia", "error_dia", "abserr_dia", "pred_sys", "error_sys", "abserr_sys" ]]
pred_data.to_csv(settings.BASE_DIR + "prediction_calibrated_" + MODEL_NAME + ".csv", sep=";")
print "Done"