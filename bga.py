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
    def __init__(self, pop_shape, pc=0.9, pm=0.005, max_round=100, chrom_l=[0, 0], low=[0, 0], high=[0, 0], selection=2, scaling=0, crossovermode=2):
        self.pop_shape = pop_shape
        self.pc = pc
        self.pm = pm
        self.max_round = max_round
        self.chrom_l = chrom_l
        self.low = low
        self.high = high
        self.selection = selection
        self.scaling = scaling
        self.crossovermode = crossovermode

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

    def nonUniformCrossoverBasedOnMask(self, population, chromosoms):
        k = int(self.pop_shape[0] / 4)
        negative_pop = []
        positive_pop = []
        negative_chrom = []
        positive_chrom = []
        pattern_mask = []
        positive_mask = []
        negative_mask = []
        new_pop = []
        ones = 0

        population_fitness = [self.fitnessFunc(
            population[i]) for i in range(0, population.shape[0])]
        sorted_pop_fitness = np.argsort(population_fitness, kind='heapsort')
        sorted_pop_fitness = sorted_pop_fitness[::-1]
        positive_pop = sorted_pop_fitness[:k]
        negative_pop = sorted_pop_fitness[self.pop_shape[0]-k:]

        for p in positive_pop:
            positive_chrom.append(chromosoms[p])

        for n in negative_pop:
            negative_chrom.append(chromosoms[n])

        for j in range(0, len(positive_chrom[0])):
            ones = 0
            for i in range(0, len(positive_chrom)):
                if positive_chrom[i][j] == 1:
                    ones += 1

            if ones > 12:
                positive_mask.append(1)
            elif ones <= 12:
                positive_mask.append(0)

        for j in range(0, len(negative_chrom[0])):
            ones = 0
            for i in range(0, len(negative_chrom)):
                if negative_chrom[i][j] == 1:
                    ones += 1

            if ones > 12:
                negative_mask.append(1)
            elif ones <= 12:
                negative_mask.append(0)

        for i in range(0, len(positive_mask)):
            if positive_mask[i] == negative_mask[i]:
                pattern_mask.append('x')
            elif positive_mask[i] != negative_mask[i]:
                pattern_mask.append(positive_mask[i])

        for i in range(0, 100):
            p1 = chromosoms[randint(0,99)]
            p2 = chromosoms[randint(0,99)]
            new_pop.append(self.checkConditions(p1, p2, pattern_mask))

        new_pop = np.asarray(new_pop, dtype=np.int32)
        return new_pop


    def checkConditions(self, p1, p2, pattern_mask):
        off = []
        for j in range(0, len(pattern_mask)):
            if p1[j] == p2[j]:
                if pattern_mask[j] == p1[j] or pattern_mask[j] == 'x':
                    off.append(p1[j])
                elif pattern_mask[j] != p1[j]:
                    if self.pfCalculator():
                        off.append(pattern_mask[j])
                    else:
                        off.append(p1[j])
            else:
                if pattern_mask[j] == 'x':
                    if self.psCalculator(p1, p2):
                        off.append(p1[j])
                    else:
                        off.append(p2[j])
                elif pattern_mask[j] == 0 or pattern_mask[j] == 1:
                    if self.pfCalculator():
                        off.append(pattern_mask[j])
                    else:
                        if self.psCalculator(p1, p2):
                            off.append(p1[j])
                        else:
                            off.append(p2[j])
        return off

    def pfCalculator(self):
        pf = 0.7
        if np.random.random_sample() < pf:
            return True
        else:
            return False

    def psCalculator(self, p1, p2):
        ps = 0
        parents = []
        parents.append(p1)
        parents.append(p2)
        parents = np.array(parents).reshape(2, 33)
        real_val = self.chromosomeDecode(parents)
        fit1 = self.fitnessFunc(real_val[0])
        fit2 = self.fitnessFunc(real_val[1])
        ps = fit1 / (fit1 + fit2)
        if np.random.random_sample() < ps:
            return True
        else:
            return False


    def multiParentCrossover(self, p1, p2, p3):
        point1 =  np.random.randint(1, len(p1))
        point2 =  np.random.randint(point1, len(p1))

        new_1 = list(np.hstack((p1[:point1], p3[point1:point2], p2[point2:])))
        new_2 = list(np.hstack((p2[:point1], p1[point1:point2], p3[point2:])))
        new_3 = list(np.hstack((p3[:point1], p2[point1:point2], p1[point2:])))

        return new_1, new_2, new_3


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

    def roulette_wheel_selection(self, population):
        chooses_ind = []
        population_fitness = sum([self.fitnessFunc(population[i])
                                 for i in range(0, population.shape[0])])
        chromosome_fitness = [self.fitnessFunc(population[i])
                              for i in range(0, population.shape[0])]

        match self.scaling:  # Which of the scales to apply
            case 0:
                # Calculate the probability of selecting each chromosome based on the fitness value
                chromosome_probabilities = [
                    chromosome_fitness[i]/population_fitness for i in range(0, len(chromosome_fitness))]
            case 1:
                scale_fitness = self.linearScaling(chromosome_fitness)
                sum_scale_fitness = sum(scale_fitness)
                chromosome_probabilities = [
                    scale_fitness[i]/sum_scale_fitness for i in range(0, len(scale_fitness))]
            case 2:
                scale_fitness = self.sigmaScaling(chromosome_fitness)
                sum_scale_fitness = sum(scale_fitness)
                chromosome_probabilities = [
                    scale_fitness[i]/sum_scale_fitness for i in range(0, len(scale_fitness))]
            case 3:
                scale_fitness = self.boltzmannSelection(chromosome_fitness)
                sum_scale_fitness = sum(scale_fitness)
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


    def selectIndMultiParent(self, chooses_ind, pop):
        new_pop = []
        for i in range(0, len(chooses_ind), 3):
            if i == 99:
                new_pop.append(list(pop[chooses_ind[i]]))
                break
            a, b, c = self.multiParentCrossover(
                pop[chooses_ind[i]], pop[chooses_ind[i+1]], pop[chooses_ind[i+2]])
            new_pop.append(a)
            new_pop.append(b)
            new_pop.append(c)
       
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

    def boltzmannSelection(self, fitness):
        new_fitness = []
        for i in range(0, self.max_round):
            new_fitness.append(fitness[i]/(self.max_round-i))
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

        for i in range(0, population.shape[0]):
            chooses_ind.append(np.random.choice([i for i in range(
                0, len(pop_probabilities))], p=pop_probabilities))  # Chromosome selection based on their probability of selection
        return chooses_ind

    def bestResult(self, population):  # calculate best fitness, avg fitness
        population_fitness = [self.fitnessFunc(population[i]) for i in range(0, population.shape[0])]
        population_best_fitness = max(population_fitness)
        agents_index = np.argmax(population_fitness)
        agents = population[agents_index]
        avg_population_fitness = sum(
            population_fitness) / len(population_fitness)
        return population_best_fitness, avg_population_fitness, population_fitness, agents

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
        agents = []
        # selection: 0 -> linearRanking; 1 -> tournamentSelection; 2 -> roulette_wheel_selection
        # scaling: 0 -> without scaling; 1 -> linearScaling; 2 -> sigmaScaling; 3 -> boltzmannSelection
        # crossover: 0 -> two points; 1-> three parents, three child; 2-> non-uniform based on mask
        ga = BGA((100, 33), chrom_l=[18, 15],
                 low=[-3, 4.1], high=[12.1, 5.8], selection=2, scaling=2, crossovermode=0)
        n_pop = ga.initialization()  # initial first population
        for i in range(0, self.max_round):
            chrom_decoded = ga.chromosomeDecode(n_pop)
            b_f, p_f, p, a = ga.bestResult(chrom_decoded)
            avg_population_fitness.append(p_f)
            population_best_fitness.append(b_f)
            population_fitness.append(p)
            agents.append(a)
            
            if ga.crossovermode == 2:
                new_child = ga.nonUniformCrossoverBasedOnMask(chrom_decoded, n_pop)
            elif ga.crossovermode == 0 or ga.crossovermode == 1:
                match ga.selection:  # Which of the selection methods to apply
                    case 0:
                        selected_ind = ga.linearRanking(chrom_decoded)
                    case 1:
                        selected_ind = ga.tournamentSelection(chrom_decoded)
                    case 2:
                        selected_ind = ga.roulette_wheel_selection(
                            chrom_decoded)
            if ga.crossovermode == 0:
                new_child = ga.selectInd(selected_ind, n_pop)
            elif ga.crossovermode == 1:
                new_child = ga.selectIndMultiParent(selected_ind, n_pop)
            new_pop = ga.mutation(new_child)
            n_pop = new_pop  # Replace the new population
        return population_best_fitness, avg_population_fitness, population_fitness, agents

    def plot(self, population_best_fitness, avg_population_fitness, population_fitness, agents):
        fig, ax = plt.subplots()
        ax.plot(avg_population_fitness, linewidth=2.0, label="avg_fitness")
        ax.plot(population_best_fitness, linewidth=2.0, label="best_fitness")
        plt.legend(loc="lower right")
        print(f"best solution: {max(population_best_fitness)}")
        print(f"best solution agents: {agents[np.argmax(population_best_fitness)]}")
        plt.show()
