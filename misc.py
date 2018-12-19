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

