from keras.models import load_model
from math import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


def get_error(prediction, point):
    distance = point - prediction
    return sqrt(sum(i*i for i in distance))


a0_35__10_81_max = np.array([-90, -80, -68, -80])
a0_42__5_71_max = np.array([-84, -69, -84, -89])
a0_73__1_58_max = np.array([-83, -86, -90, -90])
a1_26__13_8_max = np.array([-88, -90, -88, -76])
a1_57__2_7_max = np.array([-73, -79, -90, -90])
a1_66__8_45_max = np.array([-88, -88, -65, -88])
a1_83__16_19_max = np.array([-88, -90, -86, -72])
a2_3__11_56_max = np.array([-90, -81, -91, -74])
a2_75__9_81_max = np.array([-88, -79, -82, -89])
a2_82__4_6_max = np.array([-75, -91, -86, -90])

al_points = np.array([a0_35__10_81_max, a0_42__5_71_max, a0_73__1_58_max, a1_26__13_8_max, a1_57__2_7_max,
                      a1_66__8_45_max, a1_83__16_19_max, a2_3__11_56_max, a2_75__9_81_max, a2_82__4_6_max])

a0_35__10_81_min = np.array([-88, -80, -68, -79])
a0_42__5_71_min = np.array([-83, -69, -83, -88])
a0_73__1_58_min = np.array([-83, -86, -90, -90])
a1_26__13_8_min = np.array([-89, -90, -87, -75])
a1_57__2_7_min = np.array([-73, -79, -90, -90])
a1_66__8_45_min = np.array([-87, -87, -65, -88])
a1_83__16_19_min = np.array([-88, -90, -86, -71])
a2_3__11_56_min = np.array([-88, -80, -68, -74])  # VARÍA MUCHO POT 1
a2_75__9_81_min = np.array([-87, -79, -82, -87])
a2_82__4_6_min = np.array([-75, -90, -85, -90])

min_points = np.array([a0_35__10_81_min, a0_42__5_71_min, a0_73__1_58_min, a1_26__13_8_min, a1_57__2_7_min,
                       a1_66__8_45_min, a1_83__16_19_min, a2_3__11_56_min, a2_75__9_81_min, a2_82__4_6_min])

points = [np.array([0.35, 10.81]), np.array([0.42, 5.71]), np.array([0.73, 1.58]), np.array([1.26, 13.8]),
          np.array([1.57, 2.7]), np.array([1.66, 8.45]), np.array([1.83, 16.19]), np.array([2.3, 11.56]),
          np.array([2.75, 9.81]), np.array([2.82, 4.6])]


models_dir = "new_regression_models/pot1/"


def test():

    fixed_dir = models_dir + "fixed/"
    approved_dir = models_dir + "tested/approved/"
    denied_dir = models_dir + "tested/denied/"
    min_approved_dir = models_dir + "tested/min_approved/"

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
            os.rename(fixed_dir + model_name, min_approved_dir + model_name)
            print("Modelo {} guardado.".format(model_name))
        else:
            try:
                os.rename(fixed_dir + model_name, denied_dir + model_name)
            except FileExistsError:
                None
        K.clear_session()


def graph_test():
    approved_dir = models_dir + "fixed/"
    error_dir = models_dir + "error_graphs/absolute_min/"
    models = [f for f in os.listdir(approved_dir)]
    for model_name in models:
        model = load_model(approved_dir + model_name)
        value = []
        predictions = model.predict(min_points, batch_size=5)
        for pred, p in zip(predictions, points):
            value.append(get_error(pred, p))
        fig, ax = plt.subplots()
        ax.bar([i for i in range(1, 11)], value)
        ax.set(xlabel='Point', ylabel='Error (m)',
               title=model_name)
        fig.savefig(error_dir + model_name[:-3] + ".png")
        plt.close()
        K.clear_session()


graph_test()
