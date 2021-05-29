import numpy as np
from random import randint

def choose_parents(matriz, prescriptions):
    random1 = randint(0, len(matriz) - 1)
    random2 = randint(0, len(matriz) - 1)

    parent1 = matriz[random1]
    random_aux = random1

    while (matriz[random1] == matriz[random2]).all():
        random2 = randint(0, len(matriz) - 1)

    if matriz[random2][-1] > matriz[random1][-1]:
        parent1 = matriz[random2]
        random_aux = random2

    random1 = randint(0, len(matriz) - 1)
    random2 = randint(0, len(matriz) - 1)
    parent2 = matriz[random1]

    while (matriz[random1] == matriz[random2]).all() or (matriz[random_aux] == matriz[random1]).all():
        random1 = randint(0, len(matriz) - 1)

    while (matriz[random1] == matriz[random2]).all() or (matriz[random_aux] == matriz[random2]).all():
        random2 = randint(0, len(matriz) - 1)

    if matriz[random2][-1] > matriz[random1][-1]:
        parent2 = matriz[random2]

    return parent1, parent2, parent1[prescriptions:], parent2[prescriptions:]
