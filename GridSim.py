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
        """
        This loads the probability density function (joint probabilities) and turns it into a matrix
        :param filename: the name of the .csv file containing the probability density function
        """
        loaded_pdf = pandas.read_csv(filename)
        loaded_pdf = loaded_pdf.set_index("time")
        self.pdf = loaded_pdf.as_matrix()

    def simulate_jobs(self, hours):
        """
        Jobs are added if the random number generated is less than the probability of a job been added at that day and
        hour where the day and hour is calculated using a hour inputted
        :param hours: the number of hours to simulate jobs for
        """
        for i in range(hours):
            day = i % 7
            hour = i % 24
            probability = self.pdf[hour][day]
            if random.random() < probability:
                j = Job(self.all_models)
                j.randomize()
                self.queue.append(j)

    def print_job_queue(self):
        """
        A list of all the jobs that are in the queue to be simulated are printed as well as the amount of work units
        it requires and the time the job will take
        :return:
        """
        for i in range(len(self.queue)):
            job = self.queue[i]
            t = job.work_units * job.model.speed
            print("Job", i, job.model.name, job.work_units, t)

    def update(self):
        """
        This method updates the simulator's Job queue
        """
        pass

    def optimize(self):
        """
        We don't know how this will work yet.
        """
        pass


class Job:
    """
    this class determines what type of model a job is
    and creates the amount of work units it will be ?
    """

    def __init__(self, all_models):
        """
        Initializes a Job object
        :param all_models: a list of all models available
        :return: a Job object
        """
        self.all_models = all_models
        self.model = None
        self.work_units = []  # This should be a list of WorkUnit objects
        self.total_simulations = 0
        self.done = [1, 2]
        self.model_prob_list = None

    def load_model_prob(self, filename):
        """
        This method loads the probability of a specific model being selected
        :param filename: the name of the model probability file
        :return:
        """
        # TODO: The probabilities should move into the Model class rather - did this already
        # The data is loaded as a data frame
        load_model_prob = pandas.read_csv(filename)
        print(load_model_prob)
        # You can transpose the data frame
        print(load_model_prob.transpose())
        # And you can slice it by column headings
        print(self.model_prob_list["Flexi"])
        # If you don't know the columns headings you can
        print(self.model_prob_list.columns)

    def randomize(self):
        """
        A random number is generated to choose which model is used and
        A random number determines how many work units is used
        :return: nothing
        """
        index = random.randint(0, 2)
        self.model = self.all_models[index]
        self.work_units = random.randint(4000, 16000)

    def split_to_work_units(self):
        """
        This method breaks up the job into work-units
        :return: nothing
        """
        # TODO: Split the job into WorkUnit objects of equal size
        # TODO: Allocate Each Work Unit to a Computer from the list of Computer Objects
        pass

    def update(self):
        """
        This method updates the Job status
        :return:
        """
        pass


class Model:
    def __init__(self, name, speed, prob):
        """
        This method creates a Model object
        :param name: name of the model
        :param speed: speed of the model
        :param prob: probability of this model being used
        :return: a Model object
        """
        self.name = name
        self.speed = speed
        self.probability = prob


class Computer:
    def __init__(self, speed_computer):
        """
        This method creates a Computer object
        :param speed_computer: the speed of the computer
        :return: a Computer object
        """
        self.speed_computer = speed_computer
        self.work_unit_queue = []
        self.running = 0

    def add_work_unit(self, work_unit):
        """
        This method adds a work unit to a computer's queue
        :return:
        """
        pass


class WorkUnit:
    # You must pass in the parameters :)
    def __init__(self, simulations, job, computer):
        """
        This method creates a WorkUnit object
        :param simulations: the number of simulations assigned to this WorkUnit
        :param job: A Job Object so that we know who this WorkUnit belongs to
        :param computer: A Computer object so we know what Computer this WorkUnit is running on
        :return: A WorkUnit object
        """
        self.simulations = simulations
        self.job = job
        self.computer = computer
        self.sims_done = 0
        self.sims_left = simulations
        self.done = [1, 2]

    def update(self):
        """
        This method updates the WorkUnit object
        :return:
        """
        # TODO: Given that we know what computer and what Model this WorkUnit is try work out how many simulations could
        # TODO: Be done for each time period (e.g. an hour). You could use some assumptions
        # This just lets python know what type of object this is
        assert isinstance(self.computer, Computer)
        assert isinstance(self.job, Job)
        # Get the slowness of the computer and model
        slow_c = self.computer.speed_computer
        slow_m = self.job.model.speed
        # Work out how slow this WorkUnit runs
        slow_wu = 42
        # Then update the WorkUnit sims done by this amount
        pass


if __name__ == '__main__':
    models = []  # Creates an empty list of models
    # Loads the model probabilities in as a data frame
    model_prob = pandas.read_csv("Data/ModelProb.csv")
    # Loop through each model
    for m in model_prob.columns:
        models.append(Model(m, model_prob[m][1], model_prob[m][0]))

    # TODO: Create a list of Computer objects just like we have a list of Model objects
    # TODO: You can do this by constructing a file like the ModelProb.csv but with computer speed.
    # TODO: Just assume one computer's speed is 1 and the other is 0.5

    simulator = Simulator(models)
    simulator.load_pdf("Data/JointProbTotal.csv")
    simulator.simulate_jobs(2400)
    simulator.print_job_queue()
