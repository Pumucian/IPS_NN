import os
from xlrd import open_workbook
import numpy as np


def orientationToNumber(orientation):
    if "/" in orientation:
        orientation = orientation[:-1]
    orientations = ["north", "east", "south", "west"]
    return orientations.index(orientation)


# print(orientationToNumber("west/"))


# This script generates the data set for the NN to learn

def getAbsoluteMin(orientation):
    # Gets excel file names
    files = ["random_points/" + f for f in os.listdir('random_points') if '.xlsx' in f]

    max_input = []
    data = []

    orientation = orientationToNumber(orientation)

    for file in files:

        book = open_workbook(file)
        sheet = book.sheet_by_index(orientation * 6)
        point = []

        # 4 values ignoring column names and time
        for i in range(1, 5):
            frequencies = sheet.row_values(i, 1)[0:7:2]

            point.append(frequencies)

        numpy_point = np.array(point)
        max_input.append(np.amax(numpy_point, axis=0))
        data.append(numpy_point)
    return np.array(max_input)


# print(getAbsoluteMin("north/"))


def getAbsoluteMax(orientation):
    files = ["random_points/" + f for f in os.listdir('random_points') if '.xlsx' in f]

    min_input = []
    data = []
    orientation = orientationToNumber(orientation)

    for file in files:

        book = open_workbook(file)
        sheet = book.sheet_by_index(orientation*6)
        point = []

        # 4 values ignoring column names and time
        for i in range(1, 5):
            frequencies = sheet.row_values(i, 1)[0:7:2]

            point.append(frequencies)

        numpy_point = np.array(point)
        min_input.append(np.min(numpy_point, axis=0))
        data.append(numpy_point)
    return np.array(min_input)


# print(getAbsoluteMax("north/"))