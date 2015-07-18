__author__ = ""

import pandas
import random
import numpy


class Simulator:
    def __init__(self, all_models):
        self.all_models = all_models
        self.pdf = None
        self.queue = []

    def load_pdf(self, filename):
        loaded_pdf = pandas.read_csv(filename)
        loaded_pdf = loaded_pdf.set_index("time")
        self.pdf = loaded_pdf.as_matrix()

    def simulate_jobs(self, hours):
        for i in range(hours):
            day = i % 7
            hour = i % 24
            probability = self.pdf[hour][day]
            if random.random() < probability:
                j = Job(self.all_models)
                j.randomize()
                self.queue.append(j)

    def print_job_queue(self):
        for i in range(len(self.queue)):
            job = self.queue[i]
            t = job.work_units * job.model.speed
            print("Job", i, job.model.name, job.work_units, t)

    def update(self):
        pass


class Job:
    def __init__(self, all_models):
        self.all_models = all_models
        self.model = None
        self.work_units = 0

    def randomize(self):
        index = random.randint(0, 2)
        self.model = self.all_models[index]
        self.work_units = random.randint(4000, 16000)


class Model:
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed


if __name__ == '__main__':
    models = [Model("Model 1", 0.5),
              Model("Model 2", 0.75),
              Model("Model 3", 0.25)]

    simulator = Simulator(models)
    simulator.load_pdf("C:\\Users\\Tasmyn\\Documents\\BPJ\\Grid Logs\\Joint Prob Total.csv")
    simulator.simulate_jobs(2400)
    simulator.print_job_queue()