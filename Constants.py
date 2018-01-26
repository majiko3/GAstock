
# coding: utf-8

# In[17]:


import subprocess
#subprocess.run(['jupyter', 'nbconvert',
#                '--to', 'python', 'Constants.ipynb'])

#initial value
N_GENE = 8 # The number of genes.
N_IND = 100 # The number of individuals in a population.
CXPB = 0.5 # The probability of crossover. 交叉確率
MUTPB = 0.2 # The probability of individdual mutation. 個体突然変異確率
MUTINDPB = 0.05 # The probability of gene mutation. 遺伝子突然変異確率
N_GEN = 20 # The number of generation loop.

ALPHA = 0.3 #The constants in BLX-α crossover function.

TRADE_TYPE = 1 #0: onlyLong, 1: LongShort.
TRADE_NUM = 150 #The constants to determine the number of trade. #最低n日に一回取引する
