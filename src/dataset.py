#Librerias y variables globales
from src.data_cleaners import *
from openpyxl import load_workbook

##
# Estos diccionarios son para parametrizar el data frame con los valores estandar del diccnionario de datos
##

diccionario_para_columnas_y = {'SD':4,'3': 3,'1': 1,'2': 2}

columnas_a_eliminar = ["cod_mpio_o", "cod_dpto_o", "cod_mpio_r", "cod_dpto_r", "id", "semana", "uni_med_",
                       "nombre_barrio", "comuna", "tipo_ss_", "cod_mun_d", "cod_ase_", "fec_con_", "ini_sin_",
                       "evento", "year_"]

def separarGeneroEnColumnas(df_dengue_grave, df_dengue_general, nombre_columna_genero, nombre_columna_y):
    # Hago el one hot encoding
    aux_1 = pd.get_dummies(df_dengue_grave[[nombre_columna_genero]])
    aux_2 = pd.get_dummies(df_dengue_general[[nombre_columna_genero]])

    # Las añado al dataFrame
    df_dengue_grave = pd.concat([df_dengue_grave, aux_1], axis=1, join="inner")
    df_dengue_grave = df_dengue_grave.drop(columns=nombre_columna_genero)

    df_dengue_general = pd.concat([df_dengue_general, aux_2], axis=1, join="inner")
    df_dengue_general = df_dengue_general.drop(columns=nombre_columna_genero)

    # cambio las columnas para que en la última posición quede el testset del dataset
    aux_column = df_dengue_grave[nombre_columna_y]
    df_dengue_grave = df_dengue_grave.drop(columns=nombre_columna_y)
    df_dengue_grave = pd.concat([df_dengue_grave, aux_column], axis=1, join="inner")

    aux_column = df_dengue_general[nombre_columna_y]
    df_dengue_general = df_dengue_general.drop(columns=nombre_columna_y)
    df_dengue_general = pd.concat([df_dengue_general, aux_column], axis=1, join="inner")

    return df_dengue_grave, df_dengue_general

def catearLosValoresTestset(df_dengue_grave, df_dengue_general, nombre_columna_y):
    # Esto se hace porque se necesita que los valores del test sean enteros
    df_dengue_grave[nombre_columna_y] = df_dengue_grave[nombre_columna_y].astype(str)
    df_dengue_grave[nombre_columna_y] = df_dengue_grave[nombre_columna_y].apply(lambda x: x.replace(" ", ""))
    df_dengue_grave[nombre_columna_y] = df_dengue_grave[nombre_columna_y].map(diccionario_para_columnas_y)

    df_dengue_general[nombre_columna_y] = df_dengue_general[nombre_columna_y].astype(str)
    df_dengue_general[nombre_columna_y] = df_dengue_general[nombre_columna_y].apply(lambda x: x.replace(" ", ""))
    df_dengue_general[nombre_columna_y] = df_dengue_general[nombre_columna_y].map(diccionario_para_columnas_y)

    return df_dengue_grave, df_dengue_general

def guardarEnUnExcel(dataframe, nombre_excel, nombre_hoja):
    try:
        book = load_workbook('../data_base/excel/{}.xlsx'.format(nombre_excel))
        writer = pd.ExcelWriter('../data_base/excel/{}.xlsx'.format(nombre_excel), engine='xlsxwriter')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

        # Write each dataframe to a different worksheet.
        dataframe.to_excel(writer, sheet_name='{}'.format(nombre_hoja), index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
    except:
        writer = pd.ExcelWriter('../data_base/excel/{}.xlsx'.format(nombre_excel), engine='xlsxwriter')

        # Write each dataframe to a different worksheet.
        dataframe.to_excel(writer, sheet_name='{}'.format(nombre_hoja), index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


if __name__ == '__main__':
    nombre_columna_y = 'clas_dengue'
    nombre_columna_genero = 'sexo_'

    df_dengue_grave, df_dengue_general = lecturaCsv()
    df_dengue_grave = renombrarColumnas(df_dengue_grave, 'sivigila_denguegrave.')
    df_dengue_general = renombrarColumnas(df_dengue_general, 'sivigila_dengue.')
    df_dengue_grave, df_dengue_general = limpiezaPreliminar(df_dengue_grave, df_dengue_general, columnas_a_mapear)
    df_dengue_grave, df_dengue_general = eliminarColumnas(df_dengue_grave, df_dengue_general, columnas_a_eliminar)
    df_dengue_grave, df_dengue_general = separarGeneroEnColumnas(df_dengue_grave, df_dengue_general,
                                                                 nombre_columna_genero, nombre_columna_y)
    df_dengue_grave, df_dengue_general = catearLosValoresTestset(df_dengue_grave, df_dengue_general, nombre_columna_y)
    # corregir eso esta guardando solo una hoja (ERROR)
    guardarEnUnExcel(df_dengue_grave, "baseDeDatosPreliminarV1", "Dengue grave")
    #guardarEnUnExcel(df_dengue_general, "baseDeDatosPreliminar", "Dengue general")
    #Aqui se puede partir esta clase ya que se puede dajar esto como la versión 1 y para abajo otra version (OJO linea 94)
    #Ahora se van a eliminar los SDs
    df_dengue_grave, df_dengue_general = castearColumnasConSD(df_dengue_grave, df_dengue_general)
    df_dengue_grave = eliminarNulos(df_dengue_grave)
    df_dengue_general = eliminarNulos(df_dengue_general)
    # corregir eso esta guardando solo una hoja (ERROR)
    guardarEnUnExcel(df_dengue_grave, "baseDeDatosPreliminarV2", "Dengue grave")
    # guardarEnUnExcel(df_dengue_general, "baseDeDatosPreliminar", "Dengue general")
    # Aqui se puede partir esta clase ya que se puede dajar esto como la versión 2 y para abajo otra version (OJO linea 102)
    df_final = completarDataFrame(df_dengue_grave, df_dengue_general)
    guardarEnUnExcel(df_final, "datasetV1", "Casos Dengue")
    #
    df_final = cambiarValoresANumericos(df_final, columnas_a_mapear_especificas, None, diccionario_sintomas)
    guardarEnUnExcel(df_final, "datasetV2", "Casos Dengue")
    df_PO = pd.read_csv("../data_base/csv/dengue_medellin_for_svm_and_ann.csv")
    guardarEnUnExcel(df_PO, "datasetPO", "Casos Dengue")



