import csv
import numpy

csv_file = "C:\Users\Tasmyn\Documents\BPJ\Grid Logs\\log-30-CS3.csv"

new_split_file = "C:\Users\Tasmyn\Documents\BPJ\Grid Logs\\log-30-CS3 split.csv"

new_file = open(new_split_file, 'w+')

log = open(csv_file, "rb")

cnt = 0

for l in csv_file:
    with open(l) as log:
        line = log.readline()
        newline = line
        while not " " in line:
            newline += line
        new_file.write(newline)
        line = log.readline()
        cnt += 1

new_split_file.close()


