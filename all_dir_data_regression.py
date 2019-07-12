import os, csv
from xlrd import open_workbook
from statistics import mean

# This script generates the data set for the NN to learn

# Gets excel file names
files = [f for f in os.listdir('.') if '.xlsx' in f]

data = []

# Points represented in "human" alphabetic order (from 1_6 to 2_10)
point_order = [4, 0, 1, 2, 3, 9, 5, 6, 7, 8]
coord_order = [[1, 10], [1, 6], [1, 7], [1, 8], [1, 9], [2, 10], [2, 6], [2, 7], [2, 8], [2, 9]]

label_index = 0

# Data has 10 rows representing 10 ref points, each point should
# have arrays of 4 representing frequencies from 4 beacons
for file in files:

    for direction in range(0, 24, 6):

        book = open_workbook(file)
        sheet = book.sheet_by_index(direction)
        point = []

        # 100 values ignoring column names and time
        for i in range(1, 101):
            frequencies = sheet.row_values(i, 1)[0:7:2]
            # Empty data (represented as "") becomes the mean of previous entries
            while "" in frequencies:
                empty_index = frequencies.index("")
                frequencies[empty_index] = mean([row[empty_index] for row in point])

            point.append(frequencies)

        # Label provided in the point_order list since files are processed in a real alphabetic order
        label = coord_order[label_index]
        for p in point:
            p.extend(label)
            data.append(p)

    label_index += 1


with open("dataset_iBeacon_ch37_0degrees_test1_ED_IB500_r_alldirs.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
