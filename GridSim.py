__author__ = ""

import copy
import numpy
import pandas
import random
import scipy.optimize as opt
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms


class Simulator:
    def __init__(self, models, speeds, seed):
        """
        This method creates a Simulator object
        :param models: list of Model objects
        :return: a Simulator object
        """
        # A list of all available models
        self.all_models = models
        # Probability density function
        self.T = 0
        self.pdf = None
        # This is a queue of Job objects
        self.completed = []
        self.scheduled = []
        # A list of computer speeds
        self.computer_speeds = speeds
        self.not_completed = []
        self.time_over_budget = []
        self.seed = seed
        self.epsilon = 0.05

    def load_pdf(self, filename):
        """
        This loads the probability density function (joint probabilities) and turns it into a matrix
        :param filename: the name of the .csv file containing the probability density function
        """
        loaded_pdf = pandas.read_csv(filename)
        loaded_pdf = loaded_pdf.set_index("time")
        self.pdf = loaded_pdf.as_matrix()

    def simulate_jobs(self, hours, method, print_status=True):
        """
        Jobs are added if the random number generated is less than the probability of a job been added at that day and
        hour where the day and hour is calculated using a hour inputted
        :param hours: the number of hours to simulate jobs for
        """
        # Seed the random number generator
        # numpy.random.seed(self.seed)
        random.seed(self.seed)
        # Keeps track of the active job
        num_jobs = 0
        for t in range(hours):
            self.T = t
            # Calculate time and probability of adding a job
            day, hour = t % 7, t % 24
            probability = self.pdf[hour][day] * 5
            # Determine if a job is added or not
            if random.random() < probability:
                # Create a new Job object (randomly initialized)
                j = Job(self.all_models, self.computer_speeds, num_jobs, t)
                # Append to the Job schedule / queue
                self.scheduled.append(j)
                num_jobs += 1
                if method != "none" and len(self.scheduled) > 1:
                    self.optimize(method)
            self.add_fitnesses()
            # If there are jobs in the queue
            if len(self.scheduled) > 0:
                # Update the active job status
                self.scheduled[0].running = True
                self.scheduled[0].update(t)
                # If the job is completed set active to -1
                if self.scheduled[0].done:
                    # Remove the job from the queue
                    active_job = self.scheduled[0]
                    self.scheduled.remove(active_job)
                    self.completed.append(active_job)
            # Print the job queue to console
            if print_status:
                if t % 1000 == 0:
                    print("\nTesting", method, "simulation", t)
                    self.print_job_queue()
        return [self.not_completed, self.time_over_budget]

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
                print("Job ID: {0},\tModel: {1},\tSims: {2},\tDeadline time: {3},\tCompleted time : {4},\tStatus: {5}"
                      .format(str(job.ix).zfill(4), job.model.number, str(job.num_sims).zfill(7),
                              str(job.deadline).zfill(5), str(job.end).zfill(5), complete))
        # For each job in the queue
        for i in range(len(self.scheduled)):
            job = self.scheduled[i]
            complete = ""
            for wu in range(len(job.work_units)):
                complete += '%00d' % job.work_units[wu].sims_done + ":" \
                            + '%00d' % job.work_units[wu].sims + " "
            # Print out the job to the console
            print("Job ID: {0},\tModel: {1},\tSims: {2},\tDeadline time: {3},\tCompleted time : {4},\tStatus: {5}"
                  .format(str(job.ix).zfill(4), job.model.number, str(job.num_sims).zfill(7),
                          str(job.deadline).zfill(5), str(job.end).zfill(5), complete))

    def update(self, t):
        """
        This method updates the simulator's Job queue
        """
        for j in self.scheduled:
            j.update(t)

    def get_queue_fitness_deap(self, priorities):
        return -self.get_queue_fitness(priorities),

    def get_queue_fitness(self, priorities):
        """

        :param priorities:
        :return:
        """
        if type(priorities) is list:
            priorities = numpy.array(priorities)
        queue = self.order_queue(priorities)
        return self.get_fitness(queue)

    def get_ordered_queue(self, priorities):
        """

        :param priorities:
        :return:
        """
        rands = numpy.random.uniform(low=0.01, high=0.05, size=len(priorities))
        smalls = numpy.arange(0, 1.0, float(1.0/len(priorities)))
        priorities += smalls + rands
        ordered_queue = []
        indices = numpy.arange(0, len(priorities), 1)
        indices_list = list(indices)
        while len(ordered_queue) < len(self.scheduled):
            min_p, min_i = float('+inf'), -1
            for index in indices_list:
                pi = priorities[index]
                if pi <= min_p:
                    min_p = priorities[index]
                    min_i = index
            indices_list.remove(min_i)
            job_i = self.scheduled[min_i]
            ordered_queue.append(job_i)
        return ordered_queue

    def order_queue(self, priorities):
        """

        :param priorities:
        :return:
        """
        if numpy.isnan(priorities).any():
            priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
        ordered_queue, n, i = [], len(priorities), 0
        priorities_s = sorted(priorities)
        while len(ordered_queue) < len(self.scheduled):
            indices = numpy.where(priorities == priorities_s[i])[0]
            i += len(indices)
            for j in indices:
                ordered_queue.append(copy.deepcopy(self.scheduled[j]))
        return ordered_queue

    def get_fitness(self, queue, objective='time over deadline'):
        """

        :return:
        """
        not_completed = 0
        time_over_deadline = 0
        cumulative_runtime = 0
        for j in queue:
            runtime, deadline = j.get_objectives()
            cumulative_runtime += runtime
            expected = self.T + cumulative_runtime
            time_over_deadline += expected - deadline
            if expected > deadline:
                not_completed += 1
        if objective == 'All':
            return not_completed, time_over_deadline / (len(queue) + 1)
        else:
            # return pow(time_over_deadline / (len(queue) + 1), 2.0)
            return time_over_deadline

    def add_fitnesses(self):
        time_over, not_completed = 0, 0
        if len(self.completed) > 0:
            for cj in self.completed:
                time_over += cj.end - cj.deadline
                if time_over > 0:
                    not_completed += 1
            time_over /= len(self.completed)
            not_completed /= len(self.completed)
        self.time_over_budget.append(time_over)
        self.not_completed.append(not_completed)
        # not_completed, time_over_budget = self.get_fitness(self.scheduled, "All")
        # self.time_over_budget.append(time_over_budget)
        # self.not_completed.append(not_completed)

    def optimize(self, method):
        """

        :return:
        """
        # TODO: Include boundaries and constraints
        # TODO: Include maximum iterations where applicable
        # TODO: Include DEAP algorithms - genetic algorithm etc.
        if method == "scipy.basinhopping":
            # Create initial starting solution i.e. set of priorities
            priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
            # This calls the optimization algorithm and returns a result object
            # func=self.get_queue_fitness : this is the objective function
            # x0=priorities : this is the solution you start with
            res = opt.basinhopping(func=self.get_queue_fitness, x0=priorities)
            self.update_queue(res.x)
        elif method == "scipy.anneal":
            priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
            res = opt.anneal(func=self.get_queue_fitness, x0=priorities, maxiter=250, lower=0.0, upper=1.0)
            self.update_queue(res[0])
        elif method == "deap.geneticalgorithm":
            n = len(self.scheduled)
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()

            toolbox.register("attr_bool", random.uniform, 0.0, 1.0)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("mate", cxTwoPointCopy)
            toolbox.register("evaluate", self.get_queue_fitness_deap)
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)

            pop = toolbox.population(n=50)
            hof = tools.HallOfFame(1, similar=numpy.array_equal)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean)
            stats.register("min", numpy.min)
            stats.register("max", numpy.max)

            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=500,
                                               stats=stats, halloffame=hof, verbose=False)
            self.update_queue(hof[0])
        else:
            # Run default scipy.minimize function
            best_f, best_x = float('+inf'), None
            for i in range(15):
                priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
                res = opt.minimize(fun=self.get_queue_fitness, x0=priorities)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            self.update_queue(best_x)

    def update_queue(self, priorities):
        """

        :param priorities:
        :return:
        """
        optimal_queue = self.order_queue(priorities)
        self.scheduled = None
        self.scheduled = optimal_queue
        for j in self.scheduled:
            j.running = False


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
        '''
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
        '''
        if self.model.name == "ModelOne":
            power = random.randint(12, 15)
        elif self.model.name == "ModelTwo":
            power = random.randint(12, 15)
        else:
            power = random.randint(10, 13)
        self.num_sims = pow(2, power)
        """
        if power < 12:
            self.budget = 24
        elif power < 14:
            self.budget = 48
        else:
            self.budget = 72
        """
        choice = random.randint(0, 2)
        if choice == 0:
            # This is a high priority run
            self.budget = random.randint(6, 24)
        else:
            # This is a research & development run
            self.budget = random.randint(48, 120)
        self.deadline = self.start + self.budget
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
            # Update the work unit
            if not work_unit.done:
                work_unit.update()
                self.done = False
        if self.done is True:
            self.end = t
            self.runtime = self.end - self.start

    def get_expected_runtime(self):
        """
        This method returns the expected runtime of the job
        :return:
        """
        expected_run_time = 0
        for wu in self.work_units:
            expected_run_time = max(expected_run_time, wu.get_expected_time())
        return expected_run_time

    def get_objectives(self):
        return self.get_expected_runtime(), self.deadline


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

    def get_expected_time(self):
        """
        This method returns how long the work-unit should take in time units
        :return:
        """
        slow_c = self.computer_speed
        slow_m = self.job.model.speed
        slow_wu = slow_c * slow_m
        return self.sims_left / slow_wu


def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2


def mavg(data, n=3):
    fitnesses = numpy.array(data)
    ret = numpy.cumsum(fitnesses, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_experiments():
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

    methods = ["none", "scipy.basinhopping", "scipy.minimize", "scipy.anneal", "deap.geneticalgorithm"]

    name_one = "Images/Percentage-Not-Completed ("
    name_two = "Images/Hours-Over-Budget ("
    suffix = ").jpg"
    count = 1

    while count < 5:
        seed = random.randint(1000000, 1000000000)
        not_completed_results, hours_over_results = [], []
        for opt_method in methods:
            simulator = Simulator(all_models, computers, seed)
            simulator.load_pdf("Data/JointProbTotal.csv")
            result = simulator.simulate_jobs(1500, opt_method, print_status=True)
            not_completed_results.append(result[0])
            hours_over_results.append(result[1])

        for ix in range(len(not_completed_results)):
            plt.plot(mavg(not_completed_results[ix], 6), label=methods[ix])
        plt.title("Percentage of Jobs Not Completed Before Deadline")
        plt.legend(loc="best")
        plt.savefig(name_one + str(count) + suffix)
        plt.clf()
        plt.cla()
        plt.close()

        for ix in range(len(hours_over_results)):
            plt.plot(mavg(hours_over_results[ix], 6), label=methods[ix])
        plt.title("Average Hours over Deadline")
        plt.legend(loc="best")
        plt.savefig(name_two + str(count) + suffix)
        plt.clf()
        plt.cla()
        plt.close()

        count += 1


if __name__ == '__main__':
    run_experiments()
