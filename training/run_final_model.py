import pickle
import numpy as np

#filename = 'load_model/finalized_model.sav'
#dengue_type_1_x = [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#loaded_model = pickle.load(open(filename, 'rb'))

"""
filename: model's name 
Input: dengue_type_1_x
Output: result
"""
def run_model(dengue_type_1_x, model):
    result = model.predict(np.array(dengue_type_1_x))
    #print(result)
    return result

if __name__ == '__main__':
    result = loaded_model.predict(np.array(dengue_type_1_x))
    print(result)
