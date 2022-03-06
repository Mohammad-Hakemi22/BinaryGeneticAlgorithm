import numpy as np
from operator import xor
import math


class BGA():
    def __init__(self, pop_shape, fitness, pc=0.8, pm=0.01, max_round=100, chrom_l=[0, 0], low=[0, 0], high=[0, 0]):
        self.pop_shape = pop_shape
        self.method = fitness
        self.pc = pc
        self.pm = pm
        self.max_round = max_round
        self.chrom_l = chrom_l
        self.low = low
        self.high = high

    def initialization(self):
        self.pop = np.random.randint(low=0, high=2, size=self.pop_shape)

    def crossover(self, ind_0, ind_1):
        new_0, new_1 = [], []
        assert(len(ind_0) == len(ind_1))
        p_pc = np.random.random_sample(1)
        if p_pc < self.pc:
            point = np.random.randint(len(ind_0))
            new_0 = list(np.hstack((ind_0[:point], ind_1[point:])))
            new_1 = list(np.hstack((ind_1[:point], ind_0[point:])))
        else:
            new_0 = list(ind_0)
            new_1 = list(ind_1)
        assert(len(new_0) == len(new_1))

        return new_0, new_1

    def mutation(self, pop):
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

    def b2d(self, list_b):
        l = len(list_b)
        sum = 0
        for i in range(0, l):
            p = ((l-1)-i)
            sum += (pow(2, p) * list_b[i])
        return sum

    def d2r(self, b2d, lenght_b, m):
        norm = b2d/(pow(2, lenght_b) - 1)
        match m:
            case 0:
                real = self.low[0] + (norm * (self.high[0] - self.low[0]))
                return real
            case 1:
                real = self.low[1] + (norm * (self.high[1] - self.low[1]))
                return real

    def roulette_wheel_selection(self, population):
        chooses_ind = []
        population_fitness = sum([self.fitnessFunc(population[i])
                                 for i in range(0, population.shape[0])])
        chromosome_probabilities = [self.fitnessFunc(
            population[i])/population_fitness for i in range(0, population.shape[0])]
        for i in range(0, population.shape[0]):
            chooses_ind.append(np.random.choice([i for i in range(
                0, len(chromosome_probabilities))], p=chromosome_probabilities))
        return chooses_ind

    def selectInd(self, chooses_ind):
        new_pop = []
        for i in range(0, len(chooses_ind), 2):
            a, b = self.crossover(
                self.pop[chooses_ind[i]], self.pop[chooses_ind[i+1]])
            new_pop.append(a)
            new_pop.append(b)
        npa = np.asarray(new_pop, dtype=np.int32)
        return npa

    def bestResult(self, population):
        population_best_fitness = max(
            [self.fitnessFunc(population[i]) for i in range(0, population.shape[0])])
        population_fitness = [self.fitnessFunc(
            population[i]) for i in range(0, population.shape[0])]
        avg_population_fitness = sum(
            population_fitness) / len(population_fitness)
        return population_best_fitness, avg_population_fitness, population

    def run(self):
        avg_population_fitness = []
        population_best_fitness = []
        for i in range(0, self.max_round):
            x, y = 0.0, 0.0
            fitness_func = 21.5 + x*np.sin(4*np.pi*x) + y*np.sin(20*np.pi*y)
            ga = BGA((100, 33), fitness_func, chrom_l=[
                     18, 15], low=[-3, 4.1], high=[12.1, 5.8])
            ga.initialization()
            chrom_decoded = ga.chromosomeDecode()
            b_f, p_f, p = ga.bestResult(chrom_decoded)
            avg_population_fitness.append(p_f)
            population_best_fitness.append(b_f)
            selected_ind = ga.roulette_wheel_selection(chrom_decoded)
            new_child = ga.selectInd(selected_ind)
            new_pop = ga.mutation(new_child)
            self.pop = new_pop
        return population_best_fitness, avg_population_fitness
