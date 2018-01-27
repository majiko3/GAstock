
# coding: utf-8

# In[1]:


#coding: utf-8
import numpy as np
import pandas as pd
import random
import sys
#import subprocess
#subprocess.run(['jupyter', 'nbconvert',
#                '--to', 'python', 'main.ipynb'])
#以下自作ライブラリ
import realcodedGA
import AnalyseStock
from Constants import *

def main():
    #STOCK_NUM = 7974
    print('Read ****.csv (****: stock number)')
    STOCK_NUM = input('Stock number is >> ')

    f_dir = 'stock_data/'

    fr_name = f_dir + str(STOCK_NUM) + '.csv'
    fw_name = f_dir + str(STOCK_NUM) + '_new.csv'

    st = AnalyseStock.Stock()
    st.read_df(fr_name)
    df = st.set_newdf(fw_name)

    random.seed(version=2)
    # --- Create initial generation.
    pop = createPop(N_IND, N_GENE)
    fitness = setFitness(pop, df)
    best_ind, best_fit = setBest(pop, fitness)

    print('Generation loop start.')
    print('Generation: 0. Best fitness: ' + str(best_fit))
    print('Generation: 0. Best individual: ' + str(best_ind))

    #sys.exit()

    for g in range(N_GEN):
        ga = realcodedGA.Ga(pop)
        # --- Selection.
        tournsize=3
        ga.selTournament(tournsize, fitness)

        # --- Crossover.
        ga.Crossover()

        # --- Mutation.
        ga.Mutation()

        # --- Update next population.
        pop = ga.updatePop()
        pop = np.round(pop, 4)
        fitness = setFitness(pop, df)

        # --- Print best fitness in the population.
        best_ind, best_fit = setBest(pop, fitness)
        print('Generation: ' + str(g+1) + '. Best fitness: ' + str(best_fit))
        print('Generation: ' + str(g+1) + '. Best individual: ' + str(best_ind))

    print('\nGeneration loop ended. Best individual: ' + str(best_ind))

    max_dev = st.maxDev()
    min_dev = st.minDev()
    print('Entry long or short position if the signal turns on more than three quarters')
    print(
        ' \n \
        Long\n \
        RSI: less than {0}%\n \
        Psycho: less than {2}\n \
        Stochas: less than {4}%\n \
        25Dev: less than {6}%\n \
        Short\n \
        RSI: more than {1}%\n \
        Psycho: more than {3}\n \
        Stochas: more than {5}%\n \
        25Dev: more than {7}%\n　\
        '.format(round(best_ind[0]*100, 2),
                 round(best_ind[1]*100, 2),
                 best_ind[2], best_ind[3],
                 round(best_ind[4]*100, 2),
                 round(best_ind[5]*100, 2),
                 round((best_ind[6]*(max_dev-min_dev)+min_dev)*100, 2),
                 round((best_ind[7]*(max_dev-min_dev)+min_dev)*100, 2))
         )

    BackTest(fr_name, fw_name, best_ind)

    #fr_name = f_dir + str(STOCK_NUM) + '_test.csv'
    #fw_name = f_dir + str(STOCK_NUM) + '_test_new.csv'
    #BackTest(fr_name, fw_name, best_ind)


def createPop(n_ind, n_gene):
    """
    Create population in random.
    """
    pop = np.random.rand(n_ind, n_gene)
    pop = np.round(pop, 4)
    return pop


def setFitness(pop, df):
    """
    Set fitnesses of each individual in a population.
    return: List of fitness
    """
    fitness = []
    tr = AnalyseStock.Trading(df)
    for i in range(len(pop)):
        if TRADE_TYPE == 0:
            pafo_list, posi = tr.onlyLong(pop[i])
        elif TRADE_TYPE == 1:
            pafo_list, posi = tr.LongShort(pop[i])
        fit = Evaluation(pafo_list)
        fitness.append(fit)
    return fitness


def Evaluation(pafo_rate):
    """
    Evaluation　function
    ind = (rsi_long, rsi_short)
    return: fitness
    評価関数
    株式の売買パフォーマンスが良いほど高評価としたい。
    """
    fit = sum(pafo_rate) #損益率の和が大きいほど評価高い
    for i in range(len(pafo_rate)):
        draw_down = sum(pafo_rate[:i+1])
        if draw_down <= -0.20:
            fit = 0
            break
    return fit


def setBest(pop, fitness):
    """
    Return best individuals and its fitness.
    """
    best_fit = max(fitness)
    i = max(range(len(fitness)), key=lambda j: fitness[j])
    best_ind = pop[i]
    return best_ind, best_fit


def BackTest(fr_name, fw_name, best_ind):
    """
    Back test.
    """
    print('Back test start.')
    st = AnalyseStock.Stock()
    st.read_df(fr_name)
    df = st.set_newdf(fw_name)

    tr = AnalyseStock.Trading(df)
    if TRADE_TYPE == 0:
        pafo_list, posi = tr.onlyLong(best_ind)
    elif TRADE_TYPE == 1:
        pafo_list, posi = tr.LongShort(best_ind)
    fit = sum(pafo_list)
    winning_per = len([x for x in pafo_list if x>=0])/len(pafo_list)

    print('Back test result: {0}'.format(fit))
    print('The number of trade: {0}'.format(len(pafo_list)))
    print('Winning percentage: {0}%\nMax paformance: {1}\nMinimum paformance: {2}    '.format(round(winning_per*100, 2), max(pafo_list), min(pafo_list)))
    #print(pafo_list)
    #print(posi)


if __name__ == '__main__':
    main()
