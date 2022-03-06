import numpy as np


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
