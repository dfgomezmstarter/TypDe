from training import run_final_model as predictive
from prescriptive_model import run_final_model as prescriptive
import pickle
import numpy as np
import json
import sys

if __name__ == '__main__':
    filename = 'training/load_model/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    symtomps = sys.argv[1]
    symtomps = symtomps.replace('[', '').replace(']', '').split(',')
    symtomps = [[int(numeric_string) for numeric_string in symtomps]]
    """dengue_type_1_x = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    dengue_type_2_x = [[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    dengue_type_3_x = [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]"""

    predictive_model = np.concatenate((symtomps, predictive.run_model(symtomps, loaded_model)), axis=None)
    file_dengue = "data_base/npy/dengue_category_{}.npy".format(predictive_model[-1])
    matrix = np.load(file_dengue)
    prescriptive_model, avg = prescriptive.make_prescription(predictive_model, matrix)
    print(json.dumps(prescriptive_model[21:].tolist()))
    print(json.dumps(avg))
