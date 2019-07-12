import csv, os

def getData():
    files = [f for f in os.listdir('.') if '.csv' in f]
    data = []

    for file in files:

        with open(file, newline='') as f:

            reader = csv.reader(f)

            for row in reader:

                data.append([float(row[i]) if i < 4 else int(row[i]) for i in range(5)])

    return data


def getRegressionData():
    files = [f for f in os.listdir('.') if '_r.csv' in f]
    data = []

    for file in files:

        with open(file, newline='') as f:

            reader = csv.reader(f)

            for row in reader:

                data.append([float(row[i]) if i < 4 else int(row[i]) for i in range(6)])

    return data


def getAlldirsRegressionData():
    files = [f for f in os.listdir('.') if 'r_alldirs.csv' in f]
    data = []

    for file in files:

        with open(file, newline='') as f:

            reader = csv.reader(f)

            for row in reader:

                data.append([float(row[i]) if i < 4 else int(row[i]) for i in range(6)])

    return data



def getNewRegressionData(pot):
    files = [f for f in os.listdir('.') if "pot" + pot in f]
    data = []

    for file in files:

        with open(file, newline='') as f:

            reader = csv.reader(f)

            for row in reader:

                data.append([float(row[i])for i in range(6)])

    return data


def getLayersFromModelName(model):
    splitted = model[:-3].split("_")
    return [int(layer) for layer in splitted[2:]]

