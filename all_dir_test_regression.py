from keras.models import load_model
from math import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from shutil import copyfile
import get_random_points


def get_error(prediction, point):
    distance = point - prediction
    return sqrt(sum(i*i for i in distance))


# North Orientation

# a0_41__7_84_max = np.array([-69, -66, -78, -81])
# a0_42__5_71_max = np.array([-69, -66, -95, -85])
# a0_98__11_3_max = np.array([-81, -74, -71, -72])
# a1_47__6_58_max = np.array([-72, -64, -76, -93])
# a1_66__8_45_max = np.array([-72, -71, -71, -80])
# a2_2__5_34_max = np.array([-70, -63, -77, -95])
# a2_62__10_54_max = np.array([-72, -80, -58, -83])
# a2_68__8_94_max = np.array([-80, -69, -69, -77])
#
# al_points = np.array([a0_41__7_84_max, a0_42__5_71_max, a0_98__11_3_max, a1_47__6_58_max, a1_66__8_45_max,
#                       a2_2__5_34_max, a2_62__10_54_max, a2_68__8_94_max])
#
# a0_41__7_84_min = np.array([-69, -65, -77, -81])
# a0_42__5_71_min = np.array([-68, -66, -92, -84])
# a0_98__11_3_min = np.array([-81, -74, -70, -72])
# a1_47__6_58_min = np.array([-72, -64, -76, -91])
# a1_66__8_45_min = np.array([-72, -69, -70, -79])
# a2_2__5_34_min = np.array([-69, -62, -76, -94])
# a2_62__10_54_min = np.array([-72, -79, -58, -82])
# a2_68__8_94_min = np.array([-79, -68, -69, -77])
#
# min_points = np.array([a0_41__7_84_min, a0_42__5_71_min, a0_98__11_3_min, a1_47__6_58_min, a1_66__8_45_min,
#                       a2_2__5_34_min, a2_62__10_54_min, a2_68__8_94_min])
#
points = [np.array([0.41, 7.84]), np.array([0.42, 5.71]), np.array([0.98, 11.3]), np.array([1.47, 6.58]),
          np.array([1.66, 8.45]), np.array([2.2, 5.34]), np.array([2.62, 10.54]), np.array([2.68, 8.94])]


def test(orientation, al_points):
    fixed_dir = "all_dirs_models/all_models/"
    approved_dir = "all_dirs_models/" + orientation + "approved/"

    models = [f for f in os.listdir(fixed_dir)]

    for model_name in models:

        model = load_model(fixed_dir + model_name)

        keep = True

        predictions = model.predict(al_points, batch_size=3)
        for pred, p in zip(predictions, points):
            if get_error(pred, p) > 2:
                keep = False
                break
        if keep:
            copyfile(fixed_dir + model_name, approved_dir + model_name)
            print("Modelo {} guardado.".format(model_name))

        K.clear_session()


def test_min(orientation, min_points):
    fixed_dir = "all_dirs_models/all_models/"
    min_approved_dir = "all_dirs_models/" + orientation + "min_approved/"

    models = [f for f in os.listdir(fixed_dir)]

    for model_name in models:

        model = load_model(fixed_dir + model_name)

        keep = True

        predictions = model.predict(min_points, batch_size=3)
        for pred, p in zip(predictions, points):
            if get_error(pred, p) > 2:
                keep = False
                break
        if keep:
            copyfile(fixed_dir + model_name, min_approved_dir + model_name)
            print("Modelo {} guardado.".format(model_name))

        K.clear_session()


def graph_test_value(orientation, al_points):
    approved_dir = "all_dirs_models/" + orientation + "approved/"
    error_dir = "all_dirs_models/" + orientation + "error_graphs/absolute_max/"
    models = [f for f in os.listdir(approved_dir)]
    for model_name in models:
        model = load_model(approved_dir + model_name)
        value = []
        predictions = model.predict(al_points, batch_size=6)
        for pred, p in zip(predictions, points):
            value.append(get_error(pred, p))
        fig, ax = plt.subplots()
        ax.bar([i for i in range(1, 9)], value)
        ax.set(xlabel='Point', ylabel='Error (m)',
               title=model_name)
        for i, v in enumerate(value):
            ax.text(i + 0.65, v - 0.1, str(round(v, 2)), color='black', fontweight='bold')
        fig.savefig(error_dir + model_name[:-3] + ".png")
        plt.close()
        K.clear_session()


def graph_test_value_min(orientation, min_points):
    approved_dir = "all_dirs_models/" + orientation + "min_approved/"
    error_dir = "all_dirs_models/" + orientation + "error_graphs/absolute_min/"
    models = [f for f in os.listdir(approved_dir)]
    for model_name in models:
        model = load_model(approved_dir + model_name)
        value = []
        predictions = model.predict(min_points, batch_size=6)
        for pred, p in zip(predictions, points):
            value.append(get_error(pred, p))
        fig, ax = plt.subplots()
        ax.bar([i for i in range(1, 9)], value)
        ax.set(xlabel='Point', ylabel='Error (m)',
               title=model_name)
        for i, v in enumerate(value):
            ax.text(i + 0.65, v - 0.1, str(round(v, 2)), color='black', fontweight='bold')
        fig.savefig(error_dir + model_name[:-3] + ".png")
        plt.close()
        K.clear_session()


orientations = ["south/", "east/", "west/"]
for o in orientations:
    absolute_max = get_random_points.getAbsoluteMax(o)
    absolute_min = get_random_points.getAbsoluteMin(o)
    test(o, absolute_max)
    test_min(o, absolute_min)
    graph_test_value(o, absolute_max)
    graph_test_value_min(o, absolute_min)
# print(al_points)
