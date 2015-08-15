import csv

from numpy import array

             #  "C:\Users\Tasmyn\Documents\BPJ\Grid Logs\OldMutual.Alm.Models.CakeCalculator.2015-04-25.000.log",
             #  "C:\Users\Tasmyn\Documents\BPJ\Grid Logs\OldMutual.Alm.Models.CakeCalculator.2015-04-25.001.log"]
             #  "C:\Users\Tasmyn\Documents\BPJ\Grid Logs\OldMutual.Alm.Models.CakeCalculator.2015-04-26.000.log"]
# log_files = ["C:\Users\Tasmyn\Documents\BPJ\Grid Logs\OldMutual.Alm.Models.CakeCalculator.2015-04-28.000.log" ]
#  log_files = ["C:\Users\Tasmyn\Documents\BPJ\Grid Logs\OldMutual.Alm.Models.CakeCalculator.2015-04-29.000.log"]
log_files = ["C:\Users\Tasmyn\Documents\BPJ\Grid Logs\OldMutual.Alm.Models.CakeCalculator.2015-04-30.000.log"]

new_log = "C:\Users\Tasmyn\Documents\BPJ\Grid Logs\\log-30-CS3.csv"

New_file = open(new_log, 'w+')

for l in log_files:
    with open(l) as log:
        models = []
        line = log.readline()
        for i in range(1000000):
            if "CAKE string" in line:
                out = line
                while not "TRACE" in line:
                    out += line
                    line = log.readline()
                New_file.write(out)
            line = log.readline()
New_file.close()


""""
            first_split = line.split(' - ')
            second_split = first_split[0].split(' ')
            third_split = second_split[0].split('T')
            all_data = []
            all_data.extend(third_split)
            all_data.extend(second_split[1::])
            all_data.extend(first_split[1::])
            if len(all_data) == 6 and len(third_split) == 2:
                date = third_split[0]
                time = third_split[1]
                model_num = second_split[1]
                model = all_data[3].replace('[', '').replace(']', '').replace('OldMutual.Alm.', '')
                description = all_data[5].replace('.\n', '')
                if time != '' and model != '' and description != '':
                    if "Models." in model:
                        relevant_data = [date, time, model_num, model, description]
                        #  print relevant_data

                        try:
                            index = models.index(model)
                        except ValueError:
                            models.append(model)
                    s = time + "," + date + "," + model_num + "," + model + "," + description
                    New_file.write(s + "\n")
                    New_file.flush()
            line = log.readline()
        for model in models:
            print model
New_file.close()
"""""






