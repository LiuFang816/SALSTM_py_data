import helpers
import settings
import pandas
import numpy
import time
from scipy.stats import norm
import csv

MODEL_NAME = settings.MODEL_NAME
PREDICT_FILE_PATH = settings.BASE_DIR + "prediction_calibrated_" + MODEL_NAME + ".csv"
USE_CALIBRATED_PREDICTION = True
FIELD_PREFIX = "cal_" if USE_CALIBRATED_PREDICTION else ""
STD_COL = FIELD_PREFIX + "error_"


def crps_row(row, real_volume):
    p = numpy.array(row)
    v = numpy.array(range(len(p)))
    h = v >= real_volume
    sq_dists = (p - h) ** 2
    return numpy.sum(sq_dists) / float(len(sq_dists))


def make_predict_row_step(predicted_volume):
    res = []
    for i in range(0, 600):
        if predicted_volume <= i:
            res.append(1)
        else:
            res.append(0)
    return res


def make_predict_row_window(predicted_volume, window_size=12):
    res = []
    for i in range(0, 600):
        if predicted_volume <= (i - window_size):
            res.append(1)
        elif predicted_volume <= (i + window_size):
            res.append(0.5)
        else:
            res.append(0)
    return res


def make_predict_row_stdev(predicted_volume, stdev):
    res = []
    for i in range(0, 600):
        val = norm.cdf(i, loc=predicted_volume, scale=stdev)
        res.append(val)
    return res


def make_predict_row_mean_stdev(predicted_volume, mean, stdev):
    predicted_volume -= mean
    res = []
    for i in range(0, 600):
        val = norm.cdf(i, loc=predicted_volume, scale=stdev)
        val = round(val, 2)
        res.append(val)
    return res


def get_prediction_based_on_avg(age_years, sex, reference_data, field):
    reference_data = reference_data[reference_data[field] > 0]
    ok = False
    age_spread = 1
    while not ok:
        filtered_reference_data = reference_data[reference_data["age_years"] >= age_years - age_spread]
        filtered_reference_data = filtered_reference_data[filtered_reference_data["age_years"] <= age_years + age_spread]
        filtered_reference_data = filtered_reference_data[filtered_reference_data["sex"] == sex]
        age_spread += 1
        ok = len(filtered_reference_data) > 30
    pred = filtered_reference_data[field].mean()
    std = filtered_reference_data[field].std()
    return pred, std


def generate_stdevs(from_patient1, to_patient1, from_patient2, to_patient2, window_size=30):
    print "Generating stdevs"
    base_data = pandas.read_csv(PREDICT_FILE_PATH, sep=";")
    base_data = base_data[(((base_data["patient_id"] >= from_patient1) & (base_data["patient_id"] < to_patient1)) | ((base_data["patient_id"] >= from_patient2) & (base_data["patient_id"] < to_patient2))) ]
    res = base_data[["patient_id", FIELD_PREFIX + "pred_dia", STD_COL + "dia", FIELD_PREFIX + "pred_sys", STD_COL + "sys"]].copy()
    res["std_dia"] = 0
    res["std_sys"] = 0
    res["mean_dia"] = 0
    res["mean_sys"] = 0
    for dia_sys in ["dia", "sys"]:
        print dia_sys
        res = res.sort(FIELD_PREFIX + "pred_" + dia_sys)
        for index in range(0, res.shape[0]):
            start_index = max(index - window_size, 0)
            end_index = min(index + window_size, len(res) - 1)
            window_data = res.iloc[start_index:end_index].copy()
            #mx = window_data["cal_error_" + dia_sys].argmax()
            #mi = window_data["cal_error_" + dia_sys].argmin()
            #window_data = window_data.drop([mx, mi])
            window_data = window_data[window_data["patient_id"] != res.iloc[index]["patient_id"]]
            std = window_data[STD_COL + dia_sys].std()
            mean = window_data[STD_COL + dia_sys].mean()
            res.loc[res.index[index], "std_" + dia_sys] = std
            res.loc[res.index[index], "mean_" + dia_sys] = mean

    res.to_csv(settings.BASE_DIR + "prediction_errors_std.csv", sep=";")
    return res


def make_predictions(predictions_path, from_patient, to_patient, stdevs, evaluate=True, submission_file=""):
    pred_dia_col = FIELD_PREFIX + "pred_dia"
    pred_sys_col = FIELD_PREFIX + "pred_sys"
    print ""
    print "Predicting " + str(from_patient) + " - " + str(to_patient)
    org_data_frame = pandas.read_csv(predictions_path, sep=";")

    data_frame = org_data_frame[org_data_frame["patient_id"] >= from_patient]
    data_frame = data_frame[data_frame["patient_id"] <= to_patient]
    no_dia = data_frame[data_frame[pred_dia_col] == 0]
    if len(no_dia) > 0:
        print "Warning the following patients have no dia predictions : " + str(no_dia["patient_id"].values)
    no_sys = data_frame[data_frame[pred_sys_col] == 0]
    if len(no_sys) > 0:
        print "Warning the following patients have no sys predictions : " + str(no_sys["patient_id"].values)

    agg_crps_dia = []
    agg_crps_sys = []
    pred_lines = []
    headers = "Id,P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22,P23,P24,P25,P26,P27,P28,P29,P30,P31,P32,P33,P34,P35,P36,P37,P38,P39,P40,P41,P42,P43,P44,P45,P46,P47,P48,P49,P50,P51,P52,P53,P54,P55,P56,P57,P58,P59,P60,P61,P62,P63,P64,P65,P66,P67,P68,P69,P70,P71,P72,P73,P74,P75,P76,P77,P78,P79,P80,P81,P82,P83,P84,P85,P86,P87,P88,P89,P90,P91,P92,P93,P94,P95,P96,P97,P98,P99,P100,P101,P102,P103,P104,P105,P106,P107,P108,P109,P110,P111,P112,P113,P114,P115,P116,P117,P118,P119,P120,P121,P122,P123,P124,P125,P126,P127,P128,P129,P130,P131,P132,P133,P134,P135,P136,P137,P138,P139,P140,P141,P142,P143,P144,P145,P146,P147,P148,P149,P150,P151,P152,P153,P154,P155,P156,P157,P158,P159,P160,P161,P162,P163,P164,P165,P166,P167,P168,P169,P170,P171,P172,P173,P174,P175,P176,P177,P178,P179,P180,P181,P182,P183,P184,P185,P186,P187,P188,P189,P190,P191,P192,P193,P194,P195,P196,P197,P198,P199,P200,P201,P202,P203,P204,P205,P206,P207,P208,P209,P210,P211,P212,P213,P214,P215,P216,P217,P218,P219,P220,P221,P222,P223,P224,P225,P226,P227,P228,P229,P230,P231,P232,P233,P234,P235,P236,P237,P238,P239,P240,P241,P242,P243,P244,P245,P246,P247,P248,P249,P250,P251,P252,P253,P254,P255,P256,P257,P258,P259,P260,P261,P262,P263,P264,P265,P266,P267,P268,P269,P270,P271,P272,P273,P274,P275,P276,P277,P278,P279,P280,P281,P282,P283,P284,P285,P286,P287,P288,P289,P290,P291,P292,P293,P294,P295,P296,P297,P298,P299,P300,P301,P302,P303,P304,P305,P306,P307,P308,P309,P310,P311,P312,P313,P314,P315,P316,P317,P318,P319,P320,P321,P322,P323,P324,P325,P326,P327,P328,P329,P330,P331,P332,P333,P334,P335,P336,P337,P338,P339,P340,P341,P342,P343,P344,P345,P346,P347,P348,P349,P350,P351,P352,P353,P354,P355,P356,P357,P358,P359,P360,P361,P362,P363,P364,P365,P366,P367,P368,P369,P370,P371,P372,P373,P374,P375,P376,P377,P378,P379,P380,P381,P382,P383,P384,P385,P386,P387,P388,P389,P390,P391,P392,P393,P394,P395,P396,P397,P398,P399,P400,P401,P402,P403,P404,P405,P406,P407,P408,P409,P410,P411,P412,P413,P414,P415,P416,P417,P418,P419,P420,P421,P422,P423,P424,P425,P426,P427,P428,P429,P430,P431,P432,P433,P434,P435,P436,P437,P438,P439,P440,P441,P442,P443,P444,P445,P446,P447,P448,P449,P450,P451,P452,P453,P454,P455,P456,P457,P458,P459,P460,P461,P462,P463,P464,P465,P466,P467,P468,P469,P470,P471,P472,P473,P474,P475,P476,P477,P478,P479,P480,P481,P482,P483,P484,P485,P486,P487,P488,P489,P490,P491,P492,P493,P494,P495,P496,P497,P498,P499,P500,P501,P502,P503,P504,P505,P506,P507,P508,P509,P510,P511,P512,P513,P514,P515,P516,P517,P518,P519,P520,P521,P522,P523,P524,P525,P526,P527,P528,P529,P530,P531,P532,P533,P534,P535,P536,P537,P538,P539,P540,P541,P542,P543,P544,P545,P546,P547,P548,P549,P550,P551,P552,P553,P554,P555,P556,P557,P558,P559,P560,P561,P562,P563,P564,P565,P566,P567,P568,P569,P570,P571,P572,P573,P574,P575,P576,P577,P578,P579,P580,P581,P582,P583,P584,P585,P586,P587,P588,P589,P590,P591,P592,P593,P594,P595,P596,P597,P598,P599".split(",")
    pred_lines.append(headers)
    for index, row in data_frame.iterrows():
        patient_id = row["patient_id"]
        pred_dia = row[pred_dia_col]
        age_years = row["age_years"]
        real_dia = row["Diastole"]
        if evaluate and real_dia == 0:
            print "Warning real_dia 0 for patient " + str(patient_id)
        pred_sys = row[pred_sys_col]
        real_sys = row["Systole"]
        if evaluate and real_sys == 0:
            print "Warning real_sys 0 for patient " + str(patient_id)

        dia_mean = 0
        sys_mean = 0
        delta_dias = abs(stdevs[pred_dia_col] - pred_dia)
        delta_dias.sort()
        dia_index = delta_dias.index[0]
        dia_std = stdevs.loc[dia_index, "std_dia"] * 1
        # dia_mean = stdevs.loc[dia_index, "mean_dia"]

        delta_sys = abs(stdevs[pred_sys_col] - pred_sys)
        delta_sys.sort()
        sys_index = delta_sys.index[0]
        sys_std = stdevs.loc[sys_index, "std_sys"]

        slice_count = row["normal_slice_count"]
        if slice_count < 8:
            print "Patient " + str(patient_id) + " has too few slices"
            age_years = row["age_years"]
            sex = row["sex"]
            dia_mean = 0
            pred_dia, dia_std = get_prediction_based_on_avg(age_years, sex, org_data_frame, "Diastole")
            sys_mean = 0
            pred_sys, sys_std = get_prediction_based_on_avg(age_years, sex, org_data_frame, "Systole")

        pred_row_dia = make_predict_row_mean_stdev(pred_dia, dia_mean, dia_std)
        pred_row_sys = make_predict_row_mean_stdev(pred_sys, sys_mean, sys_std)
        if evaluate:
            crps_dia = crps_row(pred_row_dia, real_dia)
            crps_sys = crps_row(pred_row_sys, real_sys)
            agg_crps_dia.append(crps_dia)
            agg_crps_sys.append(crps_sys)

        dia_line = [str(patient_id) + "_Diastole"] + map(str, pred_row_dia)
        sys_line = [str(patient_id) + "_Systole"] + map(str, pred_row_sys)
        #sys_line = "501_Systole" ",".join(map(str,pred_row_sys))
        pred_lines.append(dia_line)
        pred_lines.append(sys_line)

    crps_dia = sum(agg_crps_dia) / len(agg_crps_dia)
    crps_sys = sum(agg_crps_sys) / len(agg_crps_sys)

    print "Crps dia = " + str(crps_dia)
    print "Crps sys = " + str(crps_sys)
    print "Crps agg = " + str((crps_dia + crps_sys) / 2)

    if submission_file != "":
        with open(submission_file, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(pred_lines)


if __name__ == "__main__":
    helpers.create_dir_if_not_exists(settings.BASE_DIR + "submission_files\\")
    stdevs = generate_stdevs(1, 590, 600, 700, window_size=60)  # patient 595 and 599 are corrupt
    make_predictions(PREDICT_FILE_PATH, 1, 500, stdevs)
    make_predictions(PREDICT_FILE_PATH, 501, 700, stdevs)
    make_predictions(PREDICT_FILE_PATH, 1, 700, stdevs)
    make_predictions(PREDICT_FILE_PATH, 701, 1140, stdevs, submission_file=settings.BASE_DIR + "submission_files\\submission" + MODEL_NAME + ".csv")
    print "Done , submission written to : " + settings.BASE_DIR + "submission_files\\submission" + MODEL_NAME + ".csv"
    # make_predictions("G:\\werkdata\\kaggle\\ndsb2\\prediction_data_allscale.csv", 250, 700)

