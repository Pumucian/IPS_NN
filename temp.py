from keras.models import load_model
from math import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
import get_random_points
from keras.backend import clear_session

al_points = get_random_points.getAbsoluteMax("north")

points = [np.array([0.41, 7.84]), np.array([0.42, 5.71]), np.array([0.98, 11.3]), np.array([1.47, 6.58]),
          np.array([1.66, 8.45]), np.array([2.2, 5.34]), np.array([2.62, 10.54]), np.array([2.68, 8.94])]


def get_error(prediction, point):
    distance = point - prediction
    return sqrt(sum(i*i for i in distance))


def graph_test():
    approved_dir = "regression_models/tested/approved/"
    error_dir = "regression_models/error_graphs/absolute_max_text/"
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
        clear_session()
        plt.close()


graph_test()