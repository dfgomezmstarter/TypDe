import numpy as np
from prescriptive_model.genetic_algorithm.prescriptibe_patient import find_best_combinations,cost

def find_symtomp(matrix, symtomp):
    mejoras = matrix[:, [-1]]  # Se separan las mejoras
    matrix = matrix[:,
             :-7]  # Se separan las mejoras de la matrix para que la busqueda busque arreglos con mejoras distintas
    array_to_search = np.copy(symtomp)#Aca en vez de matrix[random] se pone el arreglo que se quiere buscar tipo nd array
    # mucho ojo, el array_to_search NO puede tener la mejora
    #print(matrix.shape)
    #print(array_to_search.shape)
    same_array_indexes = np.where((matrix == array_to_search).all(axis=1))  #
    return matrix, same_array_indexes, mejoras, array_to_search

def make_prescription(symtomp, matrix):

    #file_dengue = "../data_base/npy/dengue_category_{}.npy".format(symtomp[-1])

    #matrix = np.load(file_dengue)
    matrix_complete = matrix.copy()

    matrix, same_array_indexes, mejoras, array_to_search = find_symtomp(matrix, symtomp)

    if len(same_array_indexes[0]) > 0:
        #for idx in same_array_indexes[0]:
        #    print(idx, matrix[idx], mejoras[idx][0])

        best_index = same_array_indexes[0][0]  # El primero es el mejor
        #print("Arreglo igual con la mejora mas alta:")
        #print(best_index)
        #print(matrix_complete[best_index])
        return matrix_complete[best_index]
    else:
        # No hay igual y hay que ejecutar el genetico
        mutation = 0.5
        best_answers = 1000
        population = len(matrix_complete)
        matrix = matrix_complete.copy()
        new_row = np.concatenate((symtomp, matrix[-1][22:-1]), axis=None)
        new_cost = cost(symtomp[-1], new_row)
        print("Old-> ", matrix[-1])
        matrix[-1] = np.concatenate((new_row, new_cost), axis=None)
        print("New-> ", matrix[-1])
        # posicion de las prescripciones: 21
        prescriptions = (21, len(matrix[0]) - 1)
        condition = 0
        matrix, same_array_indexes, mejoras, array_to_search = find_symtomp(matrix, symtomp)

        print(same_array_indexes)
        exit(1)

        while condition == 0:
            matrix, condition = find_best_combinations(matrix, prescriptions, mutation, best_answers, population)
            # Ahora hay que garantizar de que el areglo con nuevos sintomas este dentro de la matriz final
        pass
    #print(len(matrix))

"""if __name__ == '__main__':
    ejemplo = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ejemplo = np.array(ejemplo)
    make_prescription(ejemplo)"""