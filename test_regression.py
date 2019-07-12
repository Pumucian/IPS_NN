from keras.models import load_model
from math import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt


def get_error(prediction, point):
    distance = point - prediction
    return sqrt(sum(i*i for i in distance))

# Model 2 (4, 4, 4, 4) and 3 (8, 4, 8, 4) are pretty promising.
    # Model 4 (4, 4, 4, 4, 4) Model 5 (6, 6, 6, 6, 6)

    # a1_47__6_58 = np.array([-72, -63, -76, -91])
    # a1_47__6_58_2 = np.array([-72, -63, -76, -93])
    # a1_47__6_58_3 = np.array([-72, -63, -76, -92])

    # a1_66_8_45 = np.array([-72, -69, -71, -80])
    # a1_66_8_45_2 = np.array([-72, -71, -71, -80])
    # a1_66_8_45_3 = np.array([-72, -69, -70, -79])

    # Max for each beacon (8 first signals)


a0_41__7_84_max = np.array([-69, -66, -78, -81])
a0_42__5_71_max = np.array([-69, -66, -95, -85])
a0_98__11_3_max = np.array([-81, -74, -71, -72])
a1_47__6_58_max = np.array([-72, -64, -76, -93])
a1_66__8_45_max = np.array([-72, -71, -71, -80])
a2_2__5_34_max = np.array([-70, -63, -77, -95])
a2_62__10_54_max = np.array([-72, -80, -58, -83])
a2_68__8_94_max = np.array([-80, -69, -69, -77])

al_points = np.array([a0_41__7_84_max, a0_42__5_71_max, a0_98__11_3_max, a1_47__6_58_max, a1_66__8_45_max,
                      a2_2__5_34_max, a2_62__10_54_max, a2_68__8_94_max])

a0_41__7_84_min = np.array([-69, -65, -77, -81])
a0_42__5_71_min = np.array([-68, -66, -92, -84])
a0_98__11_3_min = np.array([-81, -74, -70, -72])
a1_47__6_58_min = np.array([-72, -64, -76, -91])
a1_66__8_45_min = np.array([-72, -69, -70, -79])
a2_2__5_34_min = np.array([-69, -62, -76, -94])
a2_62__10_54_min = np.array([-72, -79, -58, -82])
a2_68__8_94_min = np.array([-79, -68, -69, -77])

min_points = np.array([a0_41__7_84_min, a0_42__5_71_min, a0_98__11_3_min, a1_47__6_58_min, a1_66__8_45_min,
                      a2_2__5_34_min, a2_62__10_54_min, a2_68__8_94_min])

points = [np.array([0.41, 7.84]), np.array([0.42, 5.71]), np.array([0.98, 11.3]), np.array([1.47, 6.58]),
          np.array([1.66, 8.45]), np.array([2.2, 5.34]), np.array([2.62, 10.54]), np.array([2.68, 8.94])]


def test():

    fixed_dir = "regression_models/fixed/"
    approved_dir = "regression_models/tested/approved/"
    denied_dir = "regression_models/tested/denied/"

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
            os.rename(fixed_dir + model_name, approved_dir + model_name)
            print("Modelo {} guardado.".format(model_name))
        else:
            os.remove(fixed_dir + model_name)


def graph_test():
    approved_dir = "regression_models/tested/approved/"
    error_dir = "error_graphs/absolute_max/"
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
        fig.savefig(error_dir + model_name[:-3] + ".png")
        plt.close()


def min_test():

    fixed_dir = "regression_models/fixed/"
    approved_dir = "regression_models/tested/approved/"
    min_approved_dir = "regression_models/tested/min_approved/"

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
            os.rename(fixed_dir + model_name, min_approved_dir + model_name)
            print("Modelo {} guardado.".format(model_name))
        else:
            os.remove(fixed_dir + model_name)


def min_graph_test():
    approved_dir = "regression_models/tested/min_approved/"
    error_dir = "error_graphs/absolute_min/"
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
        fig.savefig(error_dir + model_name[:-3] + ".png")
        plt.close()


graph_test()
