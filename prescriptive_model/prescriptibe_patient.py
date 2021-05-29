import pandas as pd
import numpy as np
from random import random
import parents
import sons
import time

from src.dataset import guardarEnUnExcel

def read_csv(filename):
    df = pd.read_csv("../data_base/csv/{}.csv".format(filename))
    # Se cambiaron unos valores vacios por nan
    df = df.replace(r'^\s*$', np.nan, regex=True)
    #guardarEnUnExcel(df, "datasetPO_NEW", "prescriptions")
    return df

def cost(class_dengue,row):
    dic_prescriptions = {"solucion_isotonica_5-7_ml": 4, "paracetamol": 1, "beber_agua": 2, "beber_rehidratantes": 3,
                         "solucion_isotonica_5-10_ml_o_cristaloide_20_ml": 5, "hospitalizacion": 6}
    if class_dengue == 1:
        wrong_presciptions = row[dic_prescriptions["solucion_isotonica_5-7_ml"]] * 2 + row[
            dic_prescriptions["solucion_isotonica_5-10_ml_o_cristaloide_20_ml"]] * 2
        return ((np.random.randint(30, 41) * row[dic_prescriptions["paracetamol"]] + np.random.randint(50, 61) *
                                    row[dic_prescriptions["beber_agua"]]) * 0.8 ** (wrong_presciptions)) * 0.1 ** row[
                                             dic_prescriptions["hospitalizacion"]]
    elif class_dengue == 2:
        wrong_presciptions = row[dic_prescriptions["paracetamol"]] + \
                             row[dic_prescriptions["solucion_isotonica_5-10_ml_o_cristaloide_20_ml"]] * 2
        return ((np.random.randint(65, 76) * row[
            dic_prescriptions["solucion_isotonica_5-7_ml"]] +
                 np.random.randint(15, 26) * row[dic_prescriptions["beber_rehidratantes"]]) * 0.8 ** (
                 wrong_presciptions)) * 0.1 ** (1 - row[dic_prescriptions["hospitalizacion"]])
    elif class_dengue == 3:
        wrong_presciptions = row[dic_prescriptions["paracetamol"]] + row[dic_prescriptions["solucion_isotonica_5-7_ml"]] * 2
        return ((np.random.randint(65, 76) * row[
            dic_prescriptions["solucion_isotonica_5-10_ml_o_cristaloide_20_ml"]] + np.random.randint(15, 26) * row[
                                dic_prescriptions["beber_rehidratantes"]]) * 0.8 ** (wrong_presciptions)) * 0.08 ** (
                                1 - row[dic_prescriptions["hospitalizacion"]])

if __name__ == '__main__':
    #Esto esa para abrir los archivos guardados
    #with open('../data_base/npy/dengue_category_1.npy', 'rb') as f:
    #    a = np.load(f)
    #print(a.shape)
    #exit(1)

    seconds = time.time()

    general_df = read_csv('datasetPO_NEW')
    prescriptions = (list(general_df.columns).index('clas_dengue')+1, len(list(general_df.columns))-1)

    for category in range(1, 4):
        df = general_df.copy()
        df = df[df['clas_dengue'] == category]
        population = len(df)

        condition = 0
        mutation = 0.5
        best_answers = 10000

        df = df.sort_values(by=['mejora'], ascending=False)  # poblacion inicial
        print("Estoy en la categoria: ", category)
        matriz = np.array(df.values)
        while condition == 0:
            condition = 1
            parents1, parents2, prescription_parent1, prescription_parent2 = parents.choose_parents(matriz,
                                                                                                    prescriptions[0])
            son1, son2 = sons.generate_sons(prescription_parent1, prescription_parent2)
            son1 = np.concatenate((parents1[prescriptions[0] - 1], son1), axis=None)
            son2 = np.concatenate((parents2[prescriptions[0] - 1], son2), axis=None)
            son1[-1] = cost(son1[0], son1)
            son2[-1] = cost(son2[0], son2)
            random1, random2 = random(), random()
            if random1 > mutation:
                son1 = sons.mutation(son1)
                son1[-1] = cost(son1[0], son1)
            if random2 > mutation:
                son2 = sons.mutation(son2)
                son2[-1] = cost(son2[0], son2)

            son1 = np.concatenate((parents1[:prescriptions[0] - 1], son1), axis=None)
            matriz = np.vstack([matriz, np.array(son1)])
            son2 = np.concatenate((parents1[:prescriptions[0] - 1], son2), axis=None)
            matriz = np.vstack([matriz, np.array(son2)])
            matriz = matriz[(-matriz[:, len(son1) - 1]).argsort()]
            for i in range(best_answers):
                if matriz[i][-1] <= 95:
                    condition = 0
                    break
            matriz = matriz[:population]
        print("Time: ", time.time() - seconds)
        with open('../data_base/npy/dengue_category_{}.npy'.format(category), 'wb') as f:
            np.save(f, matriz[:best_answers])





