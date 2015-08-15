__author__ = 'stuart'

import os
import numpy
import pandas
from os import listdir
from os.path import isfile, join


class Extractor:
    def __init__(self, log_file_paths):
        self.log_file_paths = log_file_paths
        self.submitted_jobs = []

    def extract_jobs(self):
        count = 0
        for log_file in self.log_file_paths:
            num_lines = sum(1 for line in open(log_file))
            prev, prev2, prev3, prev4 = "", "", "", ""
            with open(log_file) as log:
                line = log.readline()
                for i in range(num_lines):
                    if "Evaluating CAKE string" in line:
                        out = line
                        while "TRACE" not in line:
                            line = log.readline()
                            if "TRACE" not in line:
                                out += line
                        self.submitted_jobs.append(out)
                        out = out.replace('\n', ' ').replace('\t', ' ').replace(' ', '.')
                        time = out[0:19]
                        mc_loc = out.find('mc.') + len('mc.')
                        mc_char = out[mc_loc]
                        mc_sims = ""
                        while mc_char != ".":
                            mc_loc += 1
                            mc_sims += mc_char
                            mc_char = out[mc_loc]
                        if "(price...(" in out:
                            model_loc = out.find("(price...(") + len("(price...(")
                        else:
                            model_loc = out.find("(cashflow...(") + len("(cashflow...(")
                        model_char = out[model_loc]
                        model = ""
                        while model_char != '.':
                            model_loc += 1
                            model += model_char
                            model_char = out[model_loc]
                        time_data = time.split('T')
                        date = time_data[0]
                        time = time_data[1]
                        date_data = date.split('-')
                        time_data = time.split(':')
                        month = date_data[1]
                        day = date_data[2]
                        hour = time_data[0]
                        if model != "assets" and model != "annuity" and "T" not in mc_sims:
                            us = model + mc_sims
                            string_out = str(count) + "," + month + "," + day + "," + hour + "," + model + "," + mc_sims
                            # print(unique_string, prev_out)
                            if us != prev and us != prev2 and us != prev3 and us != prev4:
                                print(string_out)
                                prev4 = prev3
                                prev3 = prev2
                                prev2 = prev
                                prev = us
                                count += 1
                    line = log.readline()


if __name__ == '__main__':
    log_files = []
    directories = ["/home/stuart/Documents/GitHub/Logs/OMRSRV015"]
    for directory in directories:
        dir_log_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        for log_file in dir_log_files:
            log_files.append(log_file)
    extractor = Extractor(log_files)
    extractor.extract_jobs()
