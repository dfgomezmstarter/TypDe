import numpy as np
from random import randint

def generate_sons(parents1, parents2):
    posicionCruce = randint(1, len(parents1) - 2)

    son1 = np.concatenate((parents1[0:posicionCruce], parents2[posicionCruce:-1], parents1[-1]), axis=None)
    son2 = np.concatenate((parents2[0:posicionCruce], parents1[posicionCruce:-1], parents2[-1]), axis=None)

    return son1, son2

def mutation(son):
    posicionCruce = randint(1, len(son) - 2)
    son[posicionCruce] ^= son[posicionCruce]
    return son



