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
        avg = matrix_complete[same_array_indexes, [-1]].mean()
        return matrix_complete[best_index], avg
    else:
        matrix = matrix_complete.copy()
        new_row = np.concatenate((symtomp, matrix[0][22:]), axis=None)
        return new_row, -1

    #print(len(matrix))

"""if __name__ == '__main__':
    ejemplo = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ejemplo = np.array(ejemplo)
    make_prescription(ejemplo)"""