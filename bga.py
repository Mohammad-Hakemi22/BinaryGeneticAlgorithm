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