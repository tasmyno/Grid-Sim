__author__ = ""

import pandas
import random
import numpy


class Simulator:
    def __init__(self, all_models):
        """
        This method creates a Simulator
        :param all_models: list of Model objects
        :return: a Simulator object
        """
        self.all_models = all_models
        self.pdf = None
        self.queue = []

    def load_pdf(self, filename):
        # This loads the probability density function (joint probabilities) and turns it into a matrix
        loaded_pdf = pandas.read_csv(filename)
        loaded_pdf = loaded_pdf.set_index("time")
        self.pdf = loaded_pdf.as_matrix()

    def simulate_jobs(self, hours):
        # Jobs are added if the random number generated is less
        # than the probability of a job been added at that day and hour
        # where the day and hour is calculated using a hour inputted
        for i in range(hours):
            day = i % 7
            hour = i % 24
            probability = self.pdf[hour][day]
            if random.random() < probability:
                j = Job(self.all_models)
                j.randomize()
                self.queue.append(j)

    def print_job_queue(self):
        # a list of all the jobs that are in the queue to be simulated are printed
        # as well as the amount of work units it requires and the time the job will take
        for i in range(len(self.queue)):
            job = self.queue[i]
            t = job.work_units * job.model.speed
            print("Job", i, job.model.name, job.work_units, t)

    def update(self):
        pass

    def optimize(self):
        pass


class Job:
    """
    this function determines what type of model a job is
    and creates the amount of work units it will be ?
    """
    def __init__(self, all_models):
        self.all_models = all_models
        self.model = None
        self.work_units = 0

    def randomize(self):
        # a random number is generated to choose which model is used
        # and a random number determines how many work units is used
        index = random.randint(0, 2)
        self.model = self.all_models[index]
        self.work_units = random.randint(4000, 16000)

    def split_to_work_units(self):
        pass

    def update(self):
        pass


class Model:
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed


class Computer:
    def __init__(self, speedC):
        self.speedC = speedC
        self.queueC = []
        self.running = 0

    def computer(self):
        pass


class WorkUnit:
    def __init__(self):
        self.simulations = 0
        self.job = []
        self.computer = []
        self.sims_done = 0
        self.sims_left = 0
        self.done = [1, 2]

    def work_unit(self):
        pass

    def update(self):
        pass 


if __name__ == '__main__':
    models = [Model("Model 1", 0.5),
              Model("Model 2", 0.75),
              Model("Model 3", 0.25)]

    simulator = Simulator(models)
    simulator.load_pdf("C:\\Users\\Tasmyn\\Documents\\BPJ\\Grid Logs\\Joint Prob Total.csv")
    simulator.simulate_jobs(2400)
    simulator.print_job_queue()