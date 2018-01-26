
# coding: utf-8

# In[42]:


#coding: utf-8
import numpy as np
import random
from Constants import *
import subprocess
#subprocess.run(['jupyter', 'nbconvert',
#                '--to', 'python', 'realcodedGA.ipynb'])

class Ga:

    def __init__(self, pop):
        self.pop = pop
        self.N_GENE = N_GENE
        self.N_IND = N_IND
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.MUTINDPB = MUTINDPB
        self.ALPHA = ALPHA

        self.offspring = []


    def selTournament(self, tournsize, fitness):
        """
        Selection function.
        Tournament selection.
        tournsize: tournament size
        """
        for i in range(self.N_IND):
            chosen = []
            chosen_fitness = []
            for j in range(tournsize):
                k = random.randint(0, self.N_IND-1)
                chosen.append(self.pop[k])
                chosen_fitness.append(fitness[k])
                l = max(range(len(chosen_fitness)),
                        key=lambda x: chosen_fitness[x])
            self.offspring.append(chosen[l])


    def Crossover(self):
        """
        Crossover function.
        """
        crossover = []
        for child0, child1 in zip(self.offspring[::2], self.offspring[1::2]):
            if random.random() < self.CXPB:
                child0, child1 = self.BLXalpha(child0, child1)
            crossover.append(child0)
            crossover.append(child1)
        self.offspring = crossover[:]


    def BLXalpha(self, ind0, ind1):
        """
        BLX-α crossover.
        """
        delta = []
        min_ = []
        max_ = []
        min_c = []
        max_c = []
        c0 = []
        c1 = []

        for i in range(self.N_GENE):
            delta.append(abs(ind0[i]-ind1[i]))
            min_.append(min(ind0[i],ind1[i]))
            max_.append(max(ind0[i],ind1[i]))
            min_c.append(min_[i]-self.ALPHA*delta[i])
            max_c.append(max_[i]+self.ALPHA*delta[i])
            c0.append(self.__Boundary(random.uniform(min_c[i],max_c[i]),0,1))
            c1.append(self.__Boundary(random.uniform(min_c[i],max_c[i]),0,1))
        return c0, c1


    def __Boundary(self, x, min_, max_):
        if x < min_:
            x = min_
            return x
        elif x > max_:
            x = max_
            return x
        else:
            return x


    def Mutation(self):
        """
        Mutation function.
        """
        mutant = []
        for mut in self.offspring:
            if random.random() < self.MUTPB:
                mut = self.mutFlipBit(mut)
            mutant.append(mut)
        self.offspring = mutant[:]


    def mutFlipBit(self, ind):
        mut = []
        for i in range(len(ind)):
            if random.random() < self.MUTINDPB:
                ind[i] = random.random()
            mut.append(ind[i])
        return mut


    def updatePop(self):
        return self.offspring


if __name__ == '__main__':
    print('Yes!')
