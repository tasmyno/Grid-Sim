import csv
import numpy
import pandas
from os import listdir
from os.path import isfile, join

# Script to extract raw data
raw_data = []
# The location of the log file folders
logs = ["C:\\New Grid Logs\\OMRSRV018\\"] # ,
        # "C:\\New Grid Logs\\OMRSRV018\\"]
# For each folder
for path in logs:
    # Get the log files from the folder (as a list)
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    # For each file
    for file in files:
        print("Extracting from File", file)
        # Open the log file
        with open(file) as log:
            # Empty list to store information
            models = []
            # Read the first line
            line = log.readline()
            # While the line is not empty
            while line is not "":
                try:
                    # Split the line out
                    split_dash = line.split(' - ')
                    split_space = split_dash[0].split(' ')
                    split_time = split_space[0].split('T')
                    # If we are dealing with a model then
                    if "OldMutual.Alm.Models" in split_space[2]:
                        # Get information from the split
                        date = split_time[0]
                        time_detail = split_time[1].split(':')
                        time = time_detail[0] + ":" + time_detail[1]
                        # Get the Model Name
                        model_data = split_space[2].split('.')
                        model = model_data[3]
                        raw_data.append([time, model]) # add model
                    # Read the next line
                    line = log.readline()
                except:
                    line = log.readline()
                    continue

# Turn the raw data into a pandas data frame
data_frame = pandas.DataFrame(raw_data)
data_frame.columns = ["Time", "Model"]
data_frame = data_frame.set_index("Model")

# Get all of the unique dates in the data-frame
daily_counts = data_frame.groupby(level=0).count()
print(daily_counts)
daily_counts.to_csv("Model Counts 18.csv")