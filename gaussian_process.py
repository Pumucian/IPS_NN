from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
import misc
import numpy as np
import get_random_points
import matplotlib.pyplot as plt
from joblib import dump, load

points = [np.array([0.41, 7.84]), np.array([0.42, 5.71]), np.array([0.98, 11.3]), np.array([1.47, 6.58]),
          np.array([1.66, 8.45]), np.array([2.2, 5.34]), np.array([2.62, 10.54]), np.array([2.68, 8.94])]

orientations = ["north/", "south/", "east/", "west/"]

data = np.array(misc.getAlldirsRegressionData())

np.random.shuffle(data)

X = data[:, :-2]
Y = data[:, -2:]

kernel_name = "RationalQuadratic"

squared_multiplier = 2
length_scale = 3
alpha = 0.1

kernel = squared_multiplier * RationalQuadratic(length_scale=length_scale, alpha=alpha)

print("Usando kernel {} con los parámetros {}, {}, {}".format(kernel_name, squared_multiplier, length_scale,
                                                              alpha))
model = GaussianProcessRegressor(kernel=kernel).fit(X, Y)
dump(model, "gaussian_graphs/RQ/{}_{}_{}_{}.joblib".format(kernel_name, squared_multiplier, length_scale,
                                                           alpha))
for o in orientations:
    print("Orientación {}".format(o[:-1]))
    predictions = model.predict(get_random_points.getAbsoluteMax(o))
    error = []
    for pred, p in zip(predictions, points):
        # print(pred.shape)
        error.append(np.linalg.norm(pred - p))
    fig, ax = plt.subplots()
    ax.bar([i for i in range(1, 9)], error)
    ax.set(xlabel='Point', ylabel='Error (m)',
           title="Kernel: {} Orientation: {}".format(kernel, o[:-1]))
    for ind, v in enumerate(error):
        ax.text(ind + 0.65, v - 0.1, str(round(v, 2)), color='black', fontweight='bold')
    fig.savefig("gaussian_graphs/RQ/{}_{}_{}_{}__{}.png".format(kernel_name, squared_multiplier,
                                                                length_scale, alpha, o[:-1]))
    plt.close()
    print("Gráfica guardada")

