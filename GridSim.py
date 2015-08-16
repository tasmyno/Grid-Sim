__author__ = ""

import math
import copy
import numpy
import pandas
import random
import cProfile
import scipy.optimize as opt
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms


class Simulator:
    def __init__(self, models, speeds, seed):
        """
        This method initializes a new simulator object
        :param models: a list of available models
        :param speeds: a lists of available computer speeds
        :param seed: the random number generator seed - for comparison reasons
        :return: a new Simulator object
        """
        self.seed = seed
        self.all_models = models
        self.computer_speeds = speeds

        # T keeps track of current time in the Simulation
        # pdf is the probability density function
        self.T = 0
        self.pdf = None

        # Queues of Job objects
        self.completed = []
        self.scheduled = []

        # Keeps track of performance metrics
        self.not_completed = []
        self.time_over_budget = []

        # A small value
        self.epsilon = 0.05

    def load_pdf(self, filename):
        """
        This loads the probability density function (joint probabilities) and turns it into a matrix
        :param filename: the name of the .csv file containing the probability density function
        """
        loaded_pdf = pandas.read_csv(filename)
        loaded_pdf = loaded_pdf.set_index("time")
        self.pdf = loaded_pdf.as_matrix()

    def run_simulation(self, total_simulation_hours, optimization_method,
                       print_status=False, plot_time=False, print_job_queue=False):
        """
        This method runs a simulation. Each simulation
        1) Adds jobs to the job queue using the probability density function
           - Each job has an associated model, number of monte carlo simulations
           - Each job keeps track of if it is running or not. If it is, it can be updated
           - Only one job can be running at any given time (simplifying assumption)
        2) Keeps track of which jobs have completed
           - When a job is completed i.e. all monte carlo simulations are done, it is removed from the scheduled queue
             and moved into the completed queue. Fitness metrics are based on completed jobs
        3) Optimizes the order of the scheduled queue, using optimized priorities
           - This is done using the specified method. Methods supported include:
             a. Sequential Least Squares Programming
             b. The basinhopping algorithm
             c. Simulated Annealing
             d. Genetic Algorithms

        :param total_simulation_hours: the number of hours to simulate
        :param optimization_method: the optimization method to use
        :param print_status: boolean to print out or not
        :param plot_time: boolean to print jobs by time metrics
        :return: list of lists of fitness metrics for each time step
        """
        # Seed the random number generator - done to ensure comparability between methods
        random.seed(self.seed)
        jobs_added = 0

        # For testing the model vs. real world
        jobs_added_by_hour = numpy.zeros(24)
        jobs_added_by_day = numpy.zeros(7)

        # For each time step in the number of hours
        for time_step in range(total_simulation_hours):
            self.T = time_step
            # Calculate time and probability of adding a job
            hour = time_step % 24
            day = math.floor(time_step / 24) % 7
            probability = self.pdf[hour][day] * 42
            days_passed = math.floor(time_step / 24)

            # Determine if a job is added or not
            if random.random() < probability:
                # Create a new Job object (randomly initialized)
                j = Job(self.all_models, self.computer_speeds, jobs_added, time_step)
                self.scheduled.append(j)
                # Update tracking variables
                jobs_added += 1
                jobs_added_by_day[day] += 1
                jobs_added_by_hour[hour] += 1

            # Update the schedule in some way i.e. optimize of as is
            if optimization_method != "none" and len(self.scheduled) > 1 and time_step % 6 == 0:
                # Optimize the scheduled job queue
                self.optimize(optimization_method)
            elif optimization_method == "none":
                # Implement the as-is priority system
                self.current_solution()

            # Keep track of fitness values
            self.add_fitnesses()
            # If there are jobs in the queue
            if len(self.scheduled) > 0:
                # Update the active job status
                self.scheduled[0].running = True
                self.scheduled[0].update(time_step)
                # If the job is completed set active to -1
                if self.scheduled[0].done:
                    # Remove the job from the queue
                    active_job = self.scheduled[0]
                    self.scheduled.remove(active_job)
                    self.completed.append(active_job)

            # Print the job queue to console
            if print_status:
                if time_step % 6 == 0:
                    # Print out the simulation time
                    print("Simulation time", "T =", time_step,
                          "; Days passed =", days_passed,
                          "; Day of week =", day,
                          "; Hour of day =", hour,
                          "; Optimization Method =", optimization_method)
                    print("\nTesting", optimization_method, "simulation", time_step)
                    if print_job_queue:
                        self.print_job_queue()

        # Comparison of real world to model
        if plot_time:
            self.plot_time(jobs_added_by_day, jobs_added_by_hour)

        # Return the fitness values for graphing and comparing methods
        return [self.not_completed, self.time_over_budget]

    def plot_time(self, jobs_added_by_day, jobs_added_by_hour):
        """
        This method plots the probability (percentage) of jobs added firstly by day and then by hour. This method is
        used to compare the output of the model to the real world to determine whether the simulator is an accurate
        representation of the real world or not.
        :param jobs_added_by_day: histogram of jobs added by day
        :param jobs_added_by_hour: histogram of jobs added by hour
        :return: nothing
        """
        # Plot the jobs added by day
        # These are for comparison to the real data
        plt.style.use('ggplot')
        x_axis = numpy.arange(0, 7, 1)
        jobs_by_day = numpy.array(jobs_added_by_day)
        jobs_by_day = jobs_by_day / numpy.sum(jobs_by_day)
        plt.bar(x_axis, jobs_by_day)
        plt.title("Jobs added by day")
        plt.show()
        plt.close()

        # Plot the jobs added by hour
        # These are for comparison to the real data
        plt.style.use('ggplot')
        x_axis = numpy.arange(0, 24, 1)
        jobs_by_hour = numpy.array(jobs_added_by_hour)
        jobs_by_hour = jobs_by_hour / numpy.sum(jobs_by_hour)
        plt.bar(x_axis, jobs_by_hour)
        plt.title("Jobs added by hour of day")
        plt.show()
        plt.close()

    def print_job_queue(self, verbose=True):
        """
        This method prints out the job queue to the console. If verbose is true then both completed and scheduled jobs
        are printed out to the console, otherwise only the scheduled jobs are printed out.
        :param verbose: print completed jobs?
        :return: nothing
        """
        if verbose:
            # For each job in the queue
            for i in range(len(self.completed)):
                job = self.completed[i]
                complete = "Done! Runtime = " + str(job.runtime)
                # Print the completed job to the console
                print("Job ID: {0},\tModel: {1},\tSims: {2},\tDeadline time: {3},\tCompleted time : {4},\tStatus: {5}"
                      .format(str(job.ix).zfill(4), job.model.number, str(job.mc_sims).zfill(7),
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
                  .format(str(job.ix).zfill(4), job.model.number, str(job.mc_sims).zfill(7),
                          str(job.deadline).zfill(5), str(job.end).zfill(5), complete))

    def update(self, t):
        """
        This method just updates the jobs in the scheduled queue
        :param t: the time now
        :return: nothing
        """
        for j in self.scheduled:
            j.update(t)

    def get_queue_fitness_deap(self, priorities):
        """
        This method just returns the output from the get_queue_fitness method as (A) a maximization objective - since
        the DEAP packages works best with maximization problems, and (B) a tuple because the output has to be a tuple
        :param priorities: the list of priorities for the jobs
        :return: the fitness of the job queue constructed using the fitnesses
        """
        return -self.get_queue_fitness(priorities),

    def get_queue_fitness(self, priorities):
        """
        This method creates an ordered queue using the priorities and calculates it's fitness value
        :param priorities: the list of priorities for the jobs
        :return: the fitness of the job queue constructed using the fitnesses
        """
        if type(priorities) is list:
            priorities = numpy.array(priorities)
        queue = self.order_queue(priorities)
        return self.get_fitness(queue)

    def get_ordered_queue(self, priorities):
        """
        This method constructs an ordered queue using the fitness priorities.
        DEPRECATED - this method is very computationally expensive and slow!
        :param priorities: this a list or numpy array of fitness values
        :return: a list of the jobs in the scheduled queue except that they are now ordered
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
        This method constructs an ordered queue using the fitness priorities. The difference between this method and
        the previous one is that this uses the numpy.where() method to identify the location of the nth biggest priority
        in the queue each time. This is then added, in order of priority, to the ordered queue.
        :param priorities: this a list or numpy array of fitness values
        :return: a list of the jobs in the scheduled queue except that they are now ordered
        """
        # This is just a check to make sure the algorithms didn't return any NaN priorities.
        if numpy.isnan(priorities).any():
            priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
        ordered_queue, n, i = [], len(priorities), 0
        priorities_s = sorted(priorities)
        # While the ordered queue still doesnt have all the jobs
        while len(ordered_queue) < len(self.scheduled):
            # Identify the location of the nth biggest priority in the original priority list
            indices = numpy.where(priorities == priorities_s[i])[0]
            # Update the trackers
            i += len(indices)
            for j in indices:
                # deep copy ?
                ordered_queue.append(self.scheduled[j])
        return ordered_queue

    def get_fitness(self, queue, objective='time over deadline'):
        """
        This method returns the fitness of a queue. Fitness is measured by either the total hours over the deadline of
        all completed jobs, the average of the total hours over deadline of all completed jobs, or the percentage of
        jobs which were not completed on time. These are the quantities being optimized.
        :return: the fitness value
        """
        not_completed = 0
        time_over_deadline = 0
        cumulative_runtime = 0
        # Work out how many are not completed on time
        for j in queue:
            runtime, deadline = j.get_objectives()
            cumulative_runtime += runtime
            expected = self.T + cumulative_runtime
            # Keep track of hours over the deadline
            time_over_deadline += expected - deadline
            # If not completed on time increment the counter
            if expected > deadline:
                not_completed += 1
        if objective == 'All':
            return not_completed, time_over_deadline / (len(queue) + 1)
        else:
            # return pow(time_over_deadline / (len(queue) + 1), 2.0)
            return time_over_deadline

    def add_fitnesses(self):
        """
        This method just adds fitness values to lists so that they can be graphed at the end of the simulation.
        :return: nothing
        """
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

    def optimize(self, method):
        """
        This method uses an optimization algorithm to create an optimal set of priorities.
        :return: the optimal set of priorities of the scheduled jobs
        """
        if method == "scipy.basinhopping":
            # Create initial starting solution i.e. set of priorities
            priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
            # This calls the optimization algorithm and returns a result object
            # func=self.get_queue_fitness : this is the objective function
            # x0=priorities : this is the solution you start with
            res = opt.basinhopping(func=self.get_queue_fitness, x0=priorities, niter=100)
            self.update_queue(res.x)
        elif method == "deap.geneticalgorithm":
            # The genetic algorithm works using the DEAP package
            n = len(self.scheduled)
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
            # Specify the parameters and operators for the genetic algorithm
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", random.uniform, 0.0, 1.0)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("mate", cxTwoPointCopy)
            toolbox.register("evaluate", self.get_queue_fitness_deap)
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)
            # Create the population and hall of fame containers
            pop = toolbox.population(n=150)
            hof = tools.HallOfFame(1, similar=numpy.array_equal)
            # Run the optimization algorithm to get the optimal priorities
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100,
                                               halloffame=hof, verbose=False)
            self.update_queue(hof[0])
        else:
            # Run default scipy.minimize function
            best_f, best_x = float('+inf'), None
            # 15 independent starts
            for i in range(15):
                priorities = numpy.random.uniform(low=0.0, high=1.0, size=len(self.scheduled))
                res = opt.minimize(fun=self.get_queue_fitness, x0=priorities)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            self.update_queue(best_x)

    def update_queue(self, priorities):
        """
        This method essentially updates the scheduled job queue with an ordered queue of jobs which have been ordered
        as per the priorities passed through to the method.
        :param priorities: the optimal priorities.
        :return: nothing
        """
        optimal_queue = self.order_queue(priorities)
        self.scheduled = None
        self.scheduled = optimal_queue
        for j in self.scheduled:
            j.running = False

    def current_solution(self):
        """
        This method implements the as-is solution. It gets the priorities of the jobs (between 1 and 5) from each job,
        adds it to a numpy array of priorities and updates the queue using these priorities.
        :return: nothing
        """
        priorities = numpy.zeros(len(self.scheduled))
        for i in range(len(self.scheduled)):
            priorities[i] = self.scheduled[i].priority
        self.update_queue(priorities)


class Job:
    def __init__(self, models, speeds, index, start):
        """
        Initializes a Job object
        :param models: a list of all models available
        :return: a Job object
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

        # This information was obtained from the data analysis
        self.mc_sims = 0
        # OMPP
        if self.model.name == "ModelOne":
            self.mc_sims = 65536
        # NewGen
        elif self.model.name == "ModelTwo":
            self.mc_sims = 4096
        # WPA
        elif self.model.name == "ModelThree":
            self.mc_sims = 65536
        # Flexi
        elif self.model.name == "ModelFour":
            self.mc_sims = 32768
        # Smoothie
        elif self.model.name == "ModelFive":
            self.mc_sims = 2097152
        else:
            self.mc_sims = 1024

        choice = random.randint(0, 2)
        if choice == 0:
            # This is a high priority run
            self.budget = random.randint(6, 24)
            self.priority = random.randint(4, 5)
        else:
            # This is a research & development run
            self.budget = random.randint(48, 120)
            self.priority = random.randint(1, 3)
        self.deadline = self.start + self.budget

        # Determine how many work-units to use
        self.work_units = []
        self.num_work_units = 1
        if self.mc_sims > 4096:
            self.num_work_units = int(self.mc_sims / 4096)

        # Split the job into work-units
        for i in range(self.num_work_units):
            work_unit_sims = round(self.mc_sims / self.num_work_units, 0)
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
        counter = 0
        wu_per_hour = int((self.model.speed / 4096) * 6)
        for i in range(len(self.work_units)):
            # Update the work unit
            if not self.work_units[i].done:
                self.work_units[i].update()
                self.done = False
                counter += 1
            if counter == wu_per_hour:
                i = len(self.work_units)
                break
        if self.done is True:
            self.end = t
            self.runtime = self.end - self.start

    def update_test(self, t):
        """
        This method was just an attempt to do the above method quicker. Didn't work.
        :param t:
        :return:
        """
        wu_per_hour = int((self.model.speed / 4096) * 6)
        done, i = True, 0
        while done:
            done = self.work_units[i].done
            i += 1

        start, end = i, min(i + wu_per_hour, len(self.work_units))
        # print(done, i, start, end)
        for j in range(start, end):
            self.work_units[j].sims_done = 4096
            self.work_units[j].sims_left = 0
            self.work_units[j].done = True
        self.done = True
        for j in range(len(self.work_units)):
            self.done = self.work_units[j].done
        if self.done is True:
            self.end = t
            self.runtime = self.end - self.start

    def get_expected_runtime(self):
        """
        This method returns the expected runtime of the job
        :return:
        """
        count_non_zero = 0
        expected_run_time = 0
        # Number of work units completed per hour
        wu_per_hour = int((self.model.speed / 4096) * 6)
        # For each work unit
        for wu in self.work_units:
            # Get the work-unit run time
            wu_run_time = wu.get_expected_time()
            # If the run-time is non-zero increment the still running work units
            if wu_run_time > 0:
                count_non_zero += 1
            # Keep track of the job's expected run time
            # Use the max variable because the work-units run in parallel
            expected_run_time = max(expected_run_time, wu_run_time)
        # Return the expected run-time taking into considerations work-units not yet running
        return expected_run_time * int((count_non_zero / wu_per_hour))

    def get_objectives(self):
        """
        This method just returns the result from the expected runtime and the deadline
        :return: expected run time, deadline
        """
        return self.get_expected_runtime(), self.deadline


class Model:
    def __init__(self, name, prob, speed):
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
                self.sims_left = 0
                self.sims_done = self.sims

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
    """
    This method was taken from the DEAP website. It is a faster two-point crossover implementation for the genetic
    algorithm when using numpy arrays instead of the default Python lists.
    """
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
    """
    This static method just returns a moving average of a series
    :param data: the series to "smooth out"
    :param n: the number of time steps to smooth over
    :return: the n - moving average of data
    """
    fitnesses = numpy.array(data)
    ret = numpy.cumsum(fitnesses, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_experiments():
    """
    This method actually rungs the experiments and compared the different algorithms for scheduling
    :return:
    """
    all_models = []  # Creates an empty list of models
    # Loads the model probabilities in as a data frame
    model_prob = pandas.read_csv("Data/ModelProb.csv")
    # Loop through each model
    for m in model_prob.columns:
        all_models.append(Model(m, model_prob[m][0], model_prob[m][1]))
    # Create a list of computers, half fast, half slow
    computers = []
    num_computers = 6
    for ix in range(num_computers):
        speed = 1.0
        if ix >= 3:
            speed = 2.0
        computers.append(speed)

    # The list of optimization methods available to the simulator
    # none - this implements the as-is solution
    # scipy.minimize - sequential least squared programming
    # scipy.basinhopping - the simulated annealing algorithm a.k.a basin hopping
    # deap.geneticalgorithm - the Genetic Algorithm
    methods = ["none"] #, "scipy.basinhopping", "deap.geneticalgorithm"]

    # These are just used for plotting pretty graphs
    plt.style.use("bmh")
    name_one = "Images/Percentage-Not-Completed ("
    name_two = "Images/Hours-Over-Budget ("
    suffix = ").jpg"

    # For the number of experiments to run
    count = 3
    while count < 5:
        # Generate a random seed
        seed = random.randint(1000000, 1000000000)
        not_completed_results, hours_over_results = [], []
        for opt_method in methods:
            # Simulate each optimization algorithm given the above seed
            simulator = Simulator(all_models, computers, seed)
            simulator.load_pdf("Data/JointProbTotal.csv")
            result = simulator.run_simulation(672*2, opt_method, print_status=True)
            not_completed_results.append(result[0])
            hours_over_results.append(result[1])

        # Plot the graph of the percentage of graphs not completed
        for ix in range(len(not_completed_results)):
            plt.plot(mavg(not_completed_results[ix], 6), label=methods[ix])
        plt.title("Percentage of Jobs Not Completed Before Deadline")
        plt.legend(loc="best")
        plt.savefig(name_one + str(count) + suffix)
        plt.clf()
        plt.cla()
        plt.close()

        # Plot the graph of the average number of hours over the deadline for each job
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
