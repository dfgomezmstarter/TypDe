import re
import numpy as np
import pandas as pd

diccionario_para_columnas_x = {'SD': 3,'1': 1,'2': 2}
diccionario_para_columnas_tip_cas = {'1': 1,'2': 2,'3': 3,'4':4,'5':5,'SD':6}
diccionario_para_columnas_x_sin_SD = {3:2,1:1,2:2}

columnas_a_mapear_especificas  =["acum_liquievento","caida_plaq","fiebre","cefalea","dolrretroo","malgias","artralgia",
                                 "erupcionr","dolor_abdo","vomito","somnolenci","hipotensio","hepatomeg","hem_mucosa",
                                 "hipotermia","aum_hemato","caida_plaq","acum_liquievento","extravasac","hemorr_hem",
                                 "choque","daño_organ"]

diccionario_sintomas = {"2":0, "1":1, "2.0":0, "1.0":1}
arreglo_de_columnas_faltantes_en_db_general = ['daño_organ','extravasac','hemorr_hem','choque']

columnas_a_mapear = ["tip_cas_", "acum_liquievento", "caida_plaq", "desplazami", "fiebre", "cefalea", "dolrretroo",
                     "malgias", "artralgia", "pac_hos_", "erupcionr", "dolor_abdo", "vomito", "somnolenci",
                     "hipotensio", "hepatomeg", "hem_mucosa", "hipotermia", "aum_hemato", "caida_plaq",
                     "acum_liquievento", "extravasac", "hemorr_hem", "choque", "daño_organ"]

#arreglo_de_columnas_faltantes_en_db_general = ['daño_organ', 'extravasac', 'hemorr_hem', 'choque']

def renombrarColumnas(data_frame, texto_a_eliminar):
  data_frame_final = data_frame.copy()
  for i in data_frame_final.columns:
    data_frame_final.rename(columns={i: re.sub(str(texto_a_eliminar),r'',i)}, inplace=True)
  return data_frame_final


def cambiarValoresANumericos(data_frame=None, columnas_a_mapear=None, diccionario_para_columnas_tip_cas=None,
                                        diccionario_para_columnas_x=None):
    data_frame_final = data_frame.copy()
    for i in data_frame_final.columns:
        if i in columnas_a_mapear:
            data_frame_final[i] = data_frame_final[i].astype(str)
            data_frame_final[i] = data_frame_final[i].apply(lambda x: x.replace(" ", ""))
            if i == "tip_cas_":
                data_frame_final[i] = data_frame_final[i].map(diccionario_para_columnas_tip_cas)
            else:
                data_frame_final[i] = data_frame_final[i].map(diccionario_para_columnas_x)

            # Este if es un caso en particular porque se estaba colando un valor en null en ambas columnas
            if i == "caida_plaq" or i == "acum_liquievento":
                data_frame_final[i] = data_frame_final[i].fillna(2)

            data_frame_final[i] = data_frame_final[i].astype(int)
    return data_frame_final

# Lectura de archivos
def lecturaCsv():
    df_dengue_grave = pd.read_csv("../data_base/csv/sivigila_denguegrave.csv")
    df_dengue_general = pd.read_csv("../data_base/csv/sivigila_dengue.csv")
    # Se cambiaron unos valores vacios por nan
    df_dengue_general = df_dengue_general.replace(r'^\s*$', np.nan, regex=True)
    df_dengue_grave = df_dengue_grave.replace(r'^\s*$', np.nan, regex=True)

    return df_dengue_grave, df_dengue_general

def limpiezaPreliminar(df_dengue_grave, df_dengue_general, columnas_a_mapear):
    df_dengue_general_limpia = cambiarValoresANumericos(df_dengue_general,
                                                                            columnas_a_mapear,
                                                                            diccionario_para_columnas_tip_cas,
                                                                            diccionario_para_columnas_x)
    df_dengue_grave_limpia = cambiarValoresANumericos(df_dengue_grave, columnas_a_mapear,
                                                                          diccionario_para_columnas_tip_cas,
                                                                          diccionario_para_columnas_x)
    return df_dengue_grave_limpia, df_dengue_general_limpia

def eliminarColumnas(df_dengue_grave, df_dengue_general, columnas_a_eliminar):
    # esta es la version_1
    df_dengue_grave = df_dengue_grave.copy().drop(columnas_a_eliminar, axis=1)
    # este es la version_2
    df_dengue_general = df_dengue_general.copy().drop(columnas_a_eliminar, axis=1)

    return df_dengue_grave, df_dengue_general

def castearColumnasConSD(df_dengue_grave, df_dengue_general):
    for i in columnas_a_mapear:
        if i != 'tip_cas_':
            df_dengue_grave[i] = df_dengue_grave[i].map(diccionario_para_columnas_x_sin_SD)
            if i not in arreglo_de_columnas_faltantes_en_db_general:
                df_dengue_general[i] = df_dengue_general[i].map(diccionario_para_columnas_x_sin_SD)

    return df_dengue_grave, df_dengue_general

def eliminarNulos(df):
    # solo se eliminaron las filas nulas en Tipo de dengue para la BD grande
    for i in range(len(df)):
        if df.clas_dengue[i] == 4 or ('sexo__SD' in list(df.columns) and df.sexo__SD[i] == 1):
            df = df.drop(i, axis=0)
    if 'sexo__SD' in list(df.columns):
        df = df.drop(columns=['sexo__SD'])

    return df

def completarDataFrame(df_version_3_1_pequeña,df_version_3_2_grande):
  df_final = df_version_3_2_grande.copy()
  columnas_faltantes = list(df_version_3_1_pequeña.columns)
  # agregar columnas
  for i  in columnas_faltantes:
    if(i not in list(df_final.columns)):
      df_final[i] = np.zeros(df_final.shape[0]) + 2
      # modificar filas con dengue grave
      df_final[df_final["clas_dengue"] == 3][i] = 1
  # unir las dos bases de datos
  df_final = pd.concat([df_final, df_version_3_1_pequeña])
  # poner la columna class_dengue en la ultima posición
  aux_column = df_final["clas_dengue"]
  df_final = df_final.drop(columns="clas_dengue")
  df_final = pd.concat([df_final, aux_column], axis=1, join="inner")

  df_final = df_final.reset_index()
  df_final = df_final.drop(columns=['index'])

  # solo se eliminaron las filas nulas en Tipo de dengue
  #########################################
  ## Esto se puede mejorara pero por el momento lo dejé así
  ########################################

  arreglo_eliminados = []
  for i in range(df_final.shape[0] - 1):
      if df_final.clas_dengue.values[i] == 4:
          arreglo_eliminados.append(i)
  df_final = df_final.drop(arreglo_eliminados, axis=0)

  return df_final
