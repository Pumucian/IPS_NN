from keras.models import load_model
from math import sqrt
import get_random_points
import numpy as np
import matplotlib.pyplot as plt

points = [np.array([0.41, 7.84]), np.array([0.42, 5.71]), np.array([0.98, 11.3]), np.array([1.47, 6.58]),
          np.array([1.66, 8.45]), np.array([2.2, 5.34]), np.array([2.62, 10.54]), np.array([2.68, 8.94])]


def get_error(prediction, point):
    distance = point - prediction
    return sqrt(sum(i*i for i in distance))


model = load_model("all_dirs_models/best_model_extended.h5")

orientations = ["north/", "south/", "east/", "west/"]

for o in orientations:
    min_points = get_random_points.getAbsoluteMin(o)
    value = []
    predictions = model.predict(min_points, batch_size=6)
    for pred, p in zip(predictions, points):
        value.append(get_error(pred, p))
    fig, ax = plt.subplots()
    ax.bar([i for i in range(1, 9)], value)
    ax.set(xlabel='Point', ylabel='Error (m)',
           title="Modelo extendido {}".format(o[:-1]))
    for i, v in enumerate(value):
        ax.text(i + 0.65, v - 0.1, str(round(v, 2)), color='black', fontweight='bold')
    fig.savefig("all_dirs_models/extended_model_{}".format(o[:-1]) + ".png")
    plt.close()
