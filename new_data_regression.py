import os, csv
from xlrd import open_workbook
from statistics import mean

# This script generates the data set for the NN to learn

# Gets excel file names
files_dir = "./new_data/PRPA-pot7/"
files = [f for f in os.listdir(files_dir) if f[5] == "R"]

data = []

# Points represented in "human" alphabetic order (from 1_6 to 2_10)
coord_order = [[1.6, 10.5], [1.6, 11.5], [1.6, 12.5], [1.6, 13.5], [1.6, 14.5], [1.6, 3.5], [1.6, 4.5], [1.6, 5.5],
               [1.6, 6.5], [1.6, 7.5], [1.6, 8.5], [1.6, 9.5]]

# for file, coord in zip(files, coord_order):
#     print(file, coord)

# Data has 10 rows representing 10 ref points, each point should
# have arrays of 4 representing frequencies from 4 beacons
for file in files:

    book = open_workbook(files_dir + file)
    sheet = book.sheet_by_index(0)
    point = []

    first = True

    # 100 values ignoring column names and time
    for i in range(1, 36):
        frequencies = sheet.row_values(i, 1)[0:7:2]
        # Empty data (represented as "") becomes the mean of previous entries
        for freq in frequencies:
            if freq == "":
                if first:
                    frequencies[frequencies.index(freq)] = -90.0
                else:
                    empty_index = frequencies.index("")
                    frequencies[empty_index] = mean([row[empty_index] for row in point])

        first = False
        point.append(frequencies)

    # Label provided in the point_order list since files are processed in a real alphabetic order
    label = coord_order.pop(0)
    for p in point:
        p.extend(label)
        data.append(p)


with open("dataset_newBeacons_pot7.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

