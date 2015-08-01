__author__ = ""

import pandas
import random
import numpy


class Simulator:
    def __init__(self, all_models, all_computers):
        """
        This method creates a Simulator
        :param all_models: list of Model objects
        :return: a Simulator object
        """
        self.all_models = all_models
        self.pdf = None
        # This is a queue of jobs
        self.job_queue = []
        # Added this concept
        self.completed_jobs = []
        self.all_computers = all_computers

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
                j = Job(self.all_models, self.all_computers)
                j.randomize()
                j.split_to_work_units()
                self.job_queue.append(j)
                # Added this code to test things

    def print_job_queue(self):
        """
        A list of all the jobs that are in the queue to be simulated are printed as well as the amount of work units
        it requires and the time the job will take
        :return:
        """
        for i in range(len(self.job_queue)):
            job = self.job_queue[i]
            t = job.num_work_units * job.model.speed
            print("Job", i, job.model.name, job.num_sims, t)
            for wu in range(len(job.work_units)):
                print("\tWork Unit", wu,
                      "\tSims Done", job.work_units[wu].sims_done,
                      "\tSims Left", job.work_units[wu].sims_left)

    def update(self):
        """
        This method updates the simulator's Job queue
        """
        for j in self.job_queue:
            assert isinstance(j, Job)
            j.update()
            if j.done:
                # Not tested yet.
                self.completed_jobs.append(j)
                self.job_queue.remove(j)

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
    def __init__(self, all_models, all_computers):
        """
        Initializes a Job object
        :param all_models: a list of all models available
        :return: a Job object
        """
        self.all_models = all_models
        self.model = None
        self.num_work_units = []  # This should be a list of WorkUnit objects
        self.total_simulations = 0
        self.done = False
        self.model_prob_list = None
        self.num_sims = 0
        # I see what you were trying to do here but this info should be in the WorkUnit object
        # self.new_job_name = numpy.array()
        # self.num_sims_list = numpy.array()
        self.work_units = []
        self.all_computers = all_computers
        # TODO: Add the concept of time here. I.e. when you create a new job pass a reference (an integer) which lets
        # TODO: it know when the job was created. This will allow us to track how "long" each job takes / took

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
        self.num_work_units = random.randint(4000, 16000)

    def split_to_work_units(self):
        """
        This method breaks up the job into work-units
        :return: nothing
        """
        # TODO: Split the job into WorkUnit objects of equal size
        # Changing this so that it splits this Job into work-units

        assert isinstance(self.model, Model)

        # Work out how many simulations are required for this job
        # TODO: Consider moving this to the __init__ method?

        if self.model.name == "CakeCalculator":
            self.num_sims = 11388
        elif self.model.name == "Flexi":
            self.num_sims = 87108
        elif self.model.name == "HW2GbmEtaTPathGeneration":
            self.num_sims = 2242
        elif self.model.name == "Ldwpa":
            self.num_sims = 587
        elif self.model.name == "Smoothie":
            self.num_sims = 295847
        else:
            self.num_sims = random.randint(4000, 16000)

        # If there are few sims there is one work unit

        if self.num_sims <= 4000:
            self.num_work_units = 1
        else:
            # CHanged the logic here slightly
            self.num_work_units = int(self.num_sims / 4000)

        # Now create and add work-units to the work-unit queue

        for i in range(self.num_work_units):
            work_unit_sims = round(self.num_sims / self.num_work_units, 0)
            self.work_units.append(WorkUnit(work_unit_sims, self))

        # Now allocate the unallocated work-units to computers

        for i in range(len(self.work_units)):
            computer_index = i % len(self.all_computers)
            self.work_units[i].set_computer(self.all_computers[computer_index])
            self.all_computers[computer_index].add_work_unit(self.work_units[i])

    def update(self):
        """
        This method updates the Job status
        :return:
        """
        self.done = True
        # Loop through the work units and check if they are done
        for work_unit in self.work_units:
            assert isinstance(work_unit, WorkUnit)
            # Update the work unit
            work_unit.update()
            if not work_unit.done:
                self.done = False


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
        self.available_comp = None

    def add_work_unit(self, work_unit):
        """
        This method adds a work unit to a computer's queue
        :return:
        """
        self.work_unit_queue.append(work_unit)


class WorkUnit:
    # You must pass in the parameters :)
    def __init__(self, simulations, job):
        """
        This method creates a WorkUnit object
        :param simulations: the number of simulations assigned to this WorkUnit
        :param job: A Job Object so that we know who this WorkUnit belongs to
        :return: A WorkUnit object
        """
        self.simulations = simulations
        self.job = job
        self.sims_done = 0
        self.sims_left = simulations
        self.done = [1, 2]
        self.allocated = False
        self.done = False
        self.computer = None

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

    def set_computer(self, computer):
        self.computer = computer


if __name__ == '__main__':
    models = []  # Creates an empty list of models
    # Loads the model probabilities in as a data frame
    model_prob = pandas.read_csv("Data/ModelProb.csv")
    # Loop through each model
    for m in model_prob.columns:
        models.append(Model(m, model_prob[m][1], model_prob[m][0]))

    computers = []
    num_computers = 6
    for i in range(num_computers):
        speed = 1.0
        if i >= 3:
            speed = 2.0
        computers.append(Computer(speed))

    simulator = Simulator(models, computers)
    simulator.load_pdf("Data/JointProbTotal.csv")
    simulator.simulate_jobs(2400)
    simulator.print_job_queue()
