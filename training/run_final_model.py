import pickle
import numpy as np

filename = 'load_model/finalized_model.sav'
dengue_type_1_x = [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
dengue_type_1_y = [1]
dengue_type_2_x = [[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
dengue_type_2_y = [2]
dengue_type_3_x = [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
dengue_type_3_y = [3]
loaded_model = pickle.load(open(filename, 'rb'))

"""
filename: model's name 
Input: dengue_type_1_x
Output: result
"""
if __name__ == '__main__':
    result = loaded_model.predict(np.array(dengue_type_1_x))
    print(result)