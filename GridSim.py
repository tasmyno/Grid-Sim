__author__ = ""

import numpy
import pandas
import random


class Simulator:
    def __init__(self, models, speeds):
        """
        This method creates a Simulator object
        :param models: list of Model objects
        :return: a Simulator object
        """
        # A list of all available models
        self.all_models = models
        # Probability density function
        self.pdf = None
        # This is a queue of Job objects
        self.completed = []
        self.scheduled = []
        # A list of computer speeds
        self.computer_speeds = speeds

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
        # Keeps track of the active job
        active = -1
        num_jobs = 0
        for t in range(hours):
            # Calculate time and probability of adding a job
            day, hour = t % 7, t % 24
            probability = self.pdf[hour][day]
            # Determine if a job is added or not
            if random.random() < probability:
                # Create a new Job object (randomly initialized)
                j = Job(self.all_models, self.computer_speeds, num_jobs, t)
                # Append to the Job schedule / queue
                self.scheduled.append(j)
                num_jobs += 1
                # If this is the first job we have added
                if active == -1:
                    # Make this the active job and set it to running
                    active, self.scheduled[active].running = 0, True
            # If there are jobs in the queue
            if len(self.scheduled) > 0:
                # Update the active job status
                self.scheduled[active].update(t)
                # If the job is completed set active to -1
                if self.scheduled[active].done:
                    # Remove the job from the queue
                    active_job = self.scheduled[active]
                    self.scheduled.remove(active_job)
                    self.completed.append(active_job)
                    active = -1
                # If the list of scheduled jobs isn't empty
                if len(self.scheduled) > 0:
                    active, self.scheduled[0].running = 0, True
            # Print the job queue to console
            if t % 1 == 0:
                print("\nSimulation", t, "Active Job", active)
                self.print_job_queue()

    def print_job_queue(self, verbose=True):
        """
        This method prints out the job queue to the console
        """
        if verbose:
            # For each job in the queue
            for i in range(len(self.completed)):
                job = self.completed[i]
                complete = "Done! Runtime = " + str(job.runtime)
                # Print the completed job to the console
                print("Job ID: {0},\tModel: {1},\tSims: {2},\tStatus: {3}"
                      .format(str(job.ix).zfill(4), job.model.number, str(job.num_sims).zfill(5), complete))
        # For each job in the queue
        for i in range(len(self.scheduled)):
            job = self.scheduled[i]
            complete = ""
            for wu in range(len(job.work_units)):
                complete += '%00d' % job.work_units[wu].sims_done + ":" \
                            + '%00d' % job.work_units[wu].sims + " "
            # Print out the job to the console
            print("Job ID: {0},\tModel: {1},\tSims: {2},\tStatus: {3}"
                  .format(str(job.ix).zfill(4), job.model.number, str(job.num_sims).zfill(5), complete))

    def update(self, t):
        """
        This method updates the simulator's Job queue
        """
        for j in self.scheduled:
            j.update(t)

    def optimize(self):
        """
        This method needs to reorder the job queue (scheduled) such that some objective function is optimized
        """
        pass


class Job:
    def __init__(self, models, speeds, index, start):
        """
        Initializes a Job object
        :param models: a list of all models available
        :return: a Job object

        ModelOne    - Smoothie, slow model convergence .. many simulations
        ModelTwo    - Flexi, slow model convergence .. many simulations
        ModelThree  - Cake Calculator, architecture .. few simulations
        ModelFour   - Path Generator, architecture .. few simulations
        ModelFive   - LDWPA, fast model convergence .. fewer simulations
        ModelSix    - HWGBM, architecture .. few simulations
        """
        self.ix = index
        self.models = models
        self.computer_speeds = speeds
        # Specify the model which the Job is using
        model_random = random.random()
        for m in self.models:
            self.model = m
            if model_random < m.probability:
                break
        # Keep track of "run times"
        self.end = -1
        self.runtime = -1
        self.start = start
        # Job is running if it is at the front of the queue and
        # A Job is done when all work-units are done
        self.running = False
        self.done = False
        # Determine the total number of simulations to run
        self.num_sims = 0
        if self.model.name == "ModelOne":
            self.num_sims = 131072
        elif self.model.name == "ModelTwo":
            self.num_sims = 65536
        elif self.model.name == "ModelThree":
            self.num_sims = 2048
        elif self.model.name == "ModelFour":
            self.num_sims = 2048
        elif self.model.name == "ModelFive":
            self.num_sims = 16384
        elif self.model.name == "ModelSix":
            self.num_sims = 2048
        else:
            power = random.randint(11, 16)
            self.num_sims = pow(2, power)
        # Determine how many work-units to use
        self.work_units = []
        self.num_work_units = 1
        if self.num_sims > 4096:
            self.num_work_units = int(self.num_sims / 4096)
        # Split the job into work-units
        for i in range(self.num_work_units):
            work_unit_sims = round(self.num_sims / self.num_work_units, 0)
            self.work_units.append(WorkUnit(work_unit_sims, self))
        # Split the job into work-units
        for i in range(len(self.work_units)):
            computer_index = i % len(self.computer_speeds)
            self.work_units[i].computer_speed = self.computer_speeds[computer_index]

    def update(self, t):
        """
        This method updates the Job status
        :return:
        """
        self.done = True
        # Loop through the work units and check if they are done
        for work_unit in self.work_units:
            assert isinstance(work_unit, WorkUnit)
            # Update the work unit
            if not work_unit.done:
                work_unit.update()
                self.done = False
        if self.done is True:
            self.end = t
            self.runtime = self.end - self.start

    def get_expected_runtime(self):
        return 42


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
        if self.name == "ModelOne":
            self.number = 1
        elif self.name == "ModelTwo":
            self.number = 2
        elif self.name == "ModelThree":
            self.number = 3
        elif self.name == "ModelFour":
            self.number = 4
        elif self.name == "ModelFive":
            self.number = 5
        elif self.name == "ModelSix":
            self.number = 6


class WorkUnit:
    # You must pass in the parameters :)
    def __init__(self, sims, job):
        """
        This method creates a WorkUnit object
        :param sims: the number of simulations assigned to this WorkUnit
        :param job: A Job Object so that we know who this WorkUnit belongs to
        :return: A WorkUnit object
        """
        self.job = job
        self.sims = sims
        self.sims_done = 0
        self.sims_left = sims
        self.computer_speed = 1.0
        self.done = False

    def update(self):
        """
        This method updates the WorkUnit object
        :return:
        """
        if self.job.running:
            # Get the slowness of the computer and model
            slow_c = self.computer_speed
            slow_m = self.job.model.speed
            # Work out how slow this WorkUnit runs i.e. how many sims per time step
            slow_wu = slow_c * slow_m
            self.sims_done += slow_wu
            self.sims_left -= slow_wu
            if self.sims_left <= 0.0:
                self.done = True


if __name__ == '__main__':
    all_models = []  # Creates an empty list of models
    # Loads the model probabilities in as a data frame
    model_prob = pandas.read_csv("Data/ModelProb.csv")
    # Loop through each model
    for m in model_prob.columns:
        all_models.append(Model(m, model_prob[m][1], model_prob[m][0]))

    computers = []
    num_computers = 6
    for ix in range(num_computers):
        speed = 1.0
        if ix >= 3:
            speed = 2.0
        computers.append(speed)

    simulator = Simulator(all_models, computers)
    simulator.load_pdf("Data/JointProbTotal.csv")
    simulator.simulate_jobs(2400)
