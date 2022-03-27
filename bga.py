from audioop import avg
from cProfile import label
from random import randint, random
from re import S
from xml.etree.ElementTree import PI
import numpy as np
from operator import le, xor
import math
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


class BGA():  # class binary genetic algorithm
    # initialize param
    def __init__(self, pop_shape, pc=0.9, pm=0.005, max_round=100, chrom_l=[0, 0], low=[0, 0], high=[0, 0]):
        self.pop_shape = pop_shape
        self.pc = pc
        self.pm = pm
        self.max_round = max_round
        self.chrom_l = chrom_l
        self.low = low
        self.high = high

    def initialization(self):  # initialize first population
        pop = np.random.randint(
            low=0, high=2, size=self.pop_shape)  # random number 0,1
        return pop

    def crossover(self, ind_0, ind_1):  # cross over for two individual (one point crossover)
        new_0, new_1 = [], []
        # check two individuals have same lenght
        assert(len(ind_0) == len(ind_1))
        p_pc = np.random.random_sample(1)
        if p_pc < self.pc:  # doing crossover
            point = np.random.randint(len(ind_0))
            new_0 = list(np.hstack((ind_0[:point], ind_1[point:])))
            new_1 = list(np.hstack((ind_1[:point], ind_0[point:])))
        else:  # Transfer without crossover
            new_0 = list(ind_0)
            new_1 = list(ind_1)
        # check two new childs have same lenght
        assert(len(new_0) == len(new_1))

        return new_0, new_1

    def mutation(self, pop):
        # Calculate the number of bits that must mutation
        num_mut = math.ceil(self.pm * pop.shape[0] * pop.shape[1])
        for m in range(0, num_mut):
            i = np.random.randint(0, pop.shape[0])
            j = np.random.randint(0, pop.shape[1])
            pop[i][j] = xor(pop[i][j], 1)
        return pop

    def fitnessFunc(self, real_val):
        fitness_val = 21.5 + \
            real_val[0]*np.sin(4*np.pi*real_val[0]) + \
            real_val[1]*np.sin(20*np.pi*real_val[1])
        return fitness_val

    def b2d(self, list_b):  # convert binary number to decimal number
        l = len(list_b)
        sum = 0
        for i in range(0, l):
            p = ((l-1)-i)
            sum += (pow(2, p) * list_b[i])
        return sum

    def d2r(self, b2d, lenght_b, m):  # Change the decimal number to fit in the range of problem variables
        norm = b2d/(pow(2, lenght_b) - 1)
        match m:
            case 0:
                real = self.low[0] + (norm * (self.high[0] - self.low[0]))
                return real
            case 1:
                real = self.low[1] + (norm * (self.high[1] - self.low[1]))
                return real

    # decoding the chromosome value for calculate fitness
    def chromosomeDecode(self, pop):
        gen = []
        for i in range(0, pop.shape[0]):
            l1 = pop[i][0:self.chrom_l[0]]
            l2 = pop[i][self.chrom_l[0]:]
            gen.append(self.d2r(self.b2d(list(l1)), len(l1), 0))
            gen.append(self.d2r(self.b2d(list(l2)), len(l2), 1))
        return np.array(gen).reshape(pop.shape[0], 2)

    def roulette_wheel_selection(self, population, t):
        chooses_ind = []
        population_fitness = sum([self.fitnessFunc(population[i])
                                 for i in range(0, population.shape[0])])
        p = [self.fitnessFunc(population[i])
             for i in range(0, population.shape[0])]
        # scale_fitness = self.linearScaling(p)
        # scale_fitness = self.sigmaScaling(p)
        scale_fitness = self.boltzmannSelection(p, t + 1)
        sum_scale_fitness = sum(scale_fitness)
        # Calculate the probability of selecting each chromosome based on the fitness value
        chromosome_probabilities = [
            scale_fitness[i]/sum_scale_fitness for i in range(0, len(scale_fitness))]
        for i in range(0, population.shape[0]):
            chooses_ind.append(np.random.choice([i for i in range(
                0, len(chromosome_probabilities))], p=chromosome_probabilities))  # Chromosome selection based on their probability of selection
        return chooses_ind  # return selected individuals

    def selectInd(self, chooses_ind, pop):  # Perform crossover on the selected population
        new_pop = []
        for i in range(0, len(chooses_ind), 2):
            a, b = self.crossover(
                pop[chooses_ind[i]], pop[chooses_ind[i+1]])
            new_pop.append(a)
            new_pop.append(b)
        npa = np.asarray(new_pop, dtype=np.int32)
        return npa

    def linearScaling(self, fitness):
        c = 2
        fitness_max = np.argmax(fitness)
        avg_fitness = sum(fitness) / len(fitness)
        max_fitness = max(fitness)
        fitness[fitness_max] = c * avg_fitness
        a = ((c - 1) * avg_fitness) / (max_fitness - avg_fitness)
        b = (1 - a) * avg_fitness
        scale_fitness = [((a * fit) + b) for fit in fitness]
        for i in range(len(scale_fitness)):
            if i == fitness_max:
                continue
            if scale_fitness[i] < 0:
                scale_fitness[i] = 0
        return scale_fitness

    def sigmaScaling(self, fitness):
        c = 2
        avg_fitness = sum(fitness) / len(fitness)
        standard_deviation = np.std(fitness)
        scale_fitness = [(fit - (avg_fitness - (c * standard_deviation)))
                         for fit in fitness]
        for i in range(len(scale_fitness)):
            if scale_fitness[i] < 0:
                scale_fitness[i] = 0
        return scale_fitness

    def boltzmannSelection(self, fitness, t):
        new_fitness = [fit/t for fit in fitness]
        scale_fitness = np.exp(new_fitness)
        return scale_fitness

    def linearRanking(self, population):
        chooses_ind = []
        q0 = 0.005
        q = 0.015
        population_fitness = [self.fitnessFunc(
            population[i]) for i in range(0, population.shape[0])]
        sorted_pop_fitness = np.sort(population_fitness, kind='heapsort')
        sorted_pop_fitness = sorted_pop_fitness[::-1]
        pop_probabilities = [(q - ((q - q0) * (((idx+1) - 1)/(100 - 1))))
                             for idx, val in enumerate(sorted_pop_fitness)]
        # a = sum(pop_probabilities)
        # chooses_ind = np.random.choice([i for i in range(0, len(pop_probabilities))], p=pop_probabilities)
        for i in range(0, population.shape[0]):
            chooses_ind.append(np.random.choice([i for i in range(
                0, len(pop_probabilities))], p=pop_probabilities))  # Chromosome selection based on their probability of selection
        return chooses_ind

    def bestResult(self, population):  # calculate best fitness, avg fitness
        population_best_fitness = max(
            [self.fitnessFunc(population[i]) for i in range(0, population.shape[0])])
        population_fitness = [self.fitnessFunc(
            population[i]) for i in range(0, population.shape[0])]
        avg_population_fitness = sum(
            population_fitness) / len(population_fitness)
        return population_best_fitness, avg_population_fitness, population_fitness

    def tournamentSelection(self, population):
        k = 2
        tournament_winers = []
        for t in range(self.max_round, 0, -1):
            i_idx = randint(0, population.shape[0]-1)
            j_idx = randint(0, population.shape[0]-1)
            i = population[i_idx]
            j = population[j_idx]
            if random() < 1 / (1 + np.exp(-(self.fitnessFunc(i) - self.fitnessFunc(j)) / t)):
                tournament_winers.append(i_idx)
            else:
                tournament_winers.append(i_idx)
        return tournament_winers


    def run(self):  # start algorithm
        avg_population_fitness = []
        population_best_fitness = []
        population_fitness = []
        ga = BGA((100, 33), chrom_l=[18, 15], low=[-3, 4.1], high=[12.1, 5.8])
        n_pop = ga.initialization()  # initial first population
        for i in range(0, self.max_round):
            chrom_decoded = ga.chromosomeDecode(n_pop)
            b_f, p_f, p = ga.bestResult(chrom_decoded)
            avg_population_fitness.append(p_f)
            population_best_fitness.append(b_f)
            population_fitness.append(p)
            # ch = ga.linearRanking(chrom_decoded)
            ch1 = ga.tournamentSelection(chrom_decoded)
            # selected_ind = ga.roulette_wheel_selection(chrom_decoded, i)
            new_child = ga.selectInd(ch1, n_pop)
            new_pop = ga.mutation(new_child)
            n_pop = new_pop  # Replace the new population
        return population_best_fitness, avg_population_fitness, population_fitness

    def plot(self, population_best_fitness, avg_population_fitness, population_fitness):
        fig, ax = plt.subplots()
        ax.plot(avg_population_fitness, linewidth=2.0, label="avg_fitness")
        ax.plot(population_best_fitness, linewidth=2.0, label="best_fitness")
        plt.legend(loc="lower right")
        print(f"best solution: {max(population_best_fitness)}")
        plt.show()
