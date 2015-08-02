__author__ = ""

import numpy
import pandas
import random


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
        active_job = -1
        for i in range(hours):
            day, hour = i % 7, i % 24
            probability = self.pdf[hour][day]

            if random.random() < probability:
                j = Job(self.all_models, self.all_computers)
                j.split_to_work_units()
                self.job_queue.append(j)
                if active_job == -1:
                    active_job = 0
                    self.job_queue[active_job].running = True

            if len(self.job_queue) > 0:
                self.job_queue[active_job].update()
                if self.job_queue[active_job].done:
                    active_job = -1

                for j in range(len(self.job_queue)):
                    if not self.job_queue[j].done:
                        active_job = j
                        self.job_queue[active_job].running = True
                        break

            if i % 1 == 0:
                print("\nSimulation", i, "Active Job", active_job)
                self.print_job_queue()

    def print_job_queue(self, verbose=True):
        """
        A list of all the jobs that are in the queue to be simulated are printed as well as the amount of work units
        it requires and the time the job will take
        :return:
        """
        for i in range(len(self.job_queue)):
            job = self.job_queue[i]
            if job.done:
                complete = "Done!"
            else:
                complete = ""
                for wu in range(len(job.work_units)):
                    complete += '%00d' % job.work_units[wu].sims_done + ":" \
                                + '%00d' % job.work_units[wu].simulations + " "
            if verbose:
                print("Job ID: {0},\tModel: {1},\tSims: {2},\tStatus: {3}"
                      .format(str(i).zfill(4), job.model.number, str(job.num_sims).zfill(5), complete))
            elif not job.done:
                print("Job ID: {0},\tModel: {1},\tSims: {2},\tStatus: {3}"
                      .format(str(i).zfill(4), job.model.number, str(job.num_sims).zfill(5), complete))

    def update(self):
        """
        This method updates the simulator's Job queue
        """
        for j in self.job_queue:
            assert isinstance(j, Job)
            j.update()

    def optimize(self):
        """
        We don't know how this will work yet.
        """
        pass


class Job:
    def __init__(self, all_models, all_computers):
        """
        Initializes a Job object
        :param all_models: a list of all models available
        :return: a Job object

        ModelOne    - Smoothie, slow model convergence .. many simulations
        ModelTwo    - Flexi, slow model convergence .. many simulations
        ModelThree  - Cake Calculator, architecture .. few simulations
        ModelFour   - Path Generator, architecture .. few simulations
        ModelFive   - LDWPA, fast model convergence .. fewer simulations
        ModelSix    - HWGBM, architecture .. few simulations
        """
        self.all_models = all_models
        index = random.randint(0, len(all_models) - 1)
        self.model = self.all_models[index]

        self.total_simulations = 0
        self.done = False
        self.model_prob_list = None
        self.num_sims = 0
        self.work_units = []
        self.all_computers = all_computers
        self.running = False

        if self.model.name == "ModelOne":
            self.num_sims = 65536
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

        if self.num_sims <= 4096:
            self.num_work_units = 1
        else:
            self.num_work_units = int(self.num_sims / 4096)

    def split_to_work_units(self):
        """
        This method breaks up the job into work-units
        :return: nothing
        """
        for i in range(self.num_work_units):
            work_unit_sims = round(self.num_sims / self.num_work_units, 0)
            self.work_units.append(WorkUnit(work_unit_sims, self))

        for i in range(len(self.work_units)):
            computer_index = i % len(self.all_computers)
            self.work_units[i].computer_speed = self.all_computers[computer_index]

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
            if not work_unit.done:
                work_unit.update()
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
        self.done = False
        self.allocated = False
        self.done = False
        self.computer_speed = 1.0

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
    models = []  # Creates an empty list of models
    # Loads the model probabilities in as a data frame
    model_prob = pandas.read_csv("Data/ModelProb.csv")
    # Loop through each model
    for m in model_prob.columns:
        models.append(Model(m, model_prob[m][1], model_prob[m][0]))

    computers = []
    num_computers = 6
    for ix in range(num_computers):
        speed = 1.0
        if ix >= 3:
            speed = 2.0
        computers.append(speed)

    simulator = Simulator(models, computers)
    simulator.load_pdf("Data/JointProbTotal.csv")
    simulator.simulate_jobs(2400)
