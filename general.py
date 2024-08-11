# import pandas as pd
# import numpy as np
# from scipy.linalg import eigh
# import matplotlib.pyplot as plt


# def read_xlsx(file_name: str) -> np.array:
#     df = pd.read_excel(file_name, header=None)
#     K = df.to_numpy()
#     return K


# # Leer las matrices K y M desde los archivos Excel
# K = read_xlsx("K.xlsx")
# M = read_xlsx("M.xlsx")

# # Calcular autovalores y autovectores
# autovalores, autovectores = eigh(K, M)

# # Crear DataFrames para los autovalores y autovectores
# df_autovalores = pd.DataFrame(autovalores, columns=['Autovalores'])
# df_autovectores = pd.DataFrame(autovectores)


# # Calcular autovalores y autovectores
# autovalores, autovectores = eigh(K, M)

# # Normalización de los autovectores al piso 1
# for i in range(autovectores.shape[1]):
#     # Buscar el primer valor diferente de cero en el autovector
#     for j in autovectores[:, i]:
#         if j != 0:
#             normalizador = j
#             break
#     # Normalizar todo el autovector
#     autovectores[:, i] = autovectores[:, i] / normalizador

# # Convertir a DataFrame para visualizar
# df_autovectores_normalizados = pd.DataFrame(autovectores).round(2)
# print("Autovectores normalizados al piso 1:")
# print(df_autovectores_normalizados)


# # frecuencias naturales
# frecuencias_naturales = np.sqrt(autovalores)
# frecuencias_naturales


# # periodos de vibración
# periodos_vibracion = 2 * np.pi / frecuencias_naturales
# periodos_vibracion


# import numpy as np

# # Suponiendo que 'autovectores' y 'M' ya están definidos

# # Inicialización de la variable para almacenar el factor de participación
# Li = np.zeros(autovectores.shape[1])
# influencia = [0, 0, 0, 1, 1, 1, 0, 0, 0]
# # Cálculo del factor de participación Li para cada autovector
# for i in range(autovectores.shape[1]):
#     # Cálculo del factor de participación Li
#     Li[i] = (autovectores[:, i].T @ M @ influencia)

# # Li contiene el factor de participación para cada autovector
# print("Factor de participación Li:")
# print(Li)


# # Mi=Φ'.M.Φ  (masa generalizada)
# Mi = np.zeros(autovectores.shape[1])
# for i in range(autovectores.shape[1]):
#     Mi[i] = np.dot(autovectores[:, i], np.dot(M, autovectores[:, i]))
# Mi


# # Li^2/Mi (factor masa participacion)
# Li_masa = Li ** 2 / Mi
# Li_masa


# # Li_masa es el array que contiene los factores de participación de masa (Li)

# porcentaje_masa_participativa = Li_masa / np.sum(Li_masa)

# porcentaje_masa_participativa


# import numpy as np


# def calcular_espectro_E030_PERU(Tn, Z, S, R, U, Tp, Tl):
#     """
#     Calcula el espectro de diseño para una lista de periodos de vibración.

#     Parameters:
#     - Tn: Lista de periodos de vibración.
#     - Z: Zona sísmica.
#     - S: Factor de tipo de suelo.
#     - R: Coeficiente de reducción de fuerzas sísmicas.
#     - U: Factor de uso.
#     - Tp: Periodo de corte.
#     - Tl: Periodo límite.

#     Returns:
#     - Lista de espectros de diseño.
#     """
#     # Cálculo del espectro base
#     espectro_base = (Z * U * S) / R

#     # Calcular espectro para cada periodo de vibración
#     espectro = []
#     for tn in Tn:
#         if tn < Tp:
#             C = 2.5
#         elif Tp <= tn <= Tl:
#             C = 2.5 * (Tp / tn)
#         else:  # tn > Tl
#             C = 2.5 * ((Tp * Tl) / tn**2)

#         espectro_value = espectro_base * C * 981
#         espectro.append(espectro_value)

#     return np.array(espectro)


# # Valores proporcionados
# Z = 0.25
# S = 1.2
# R = 8
# U = 1
# Tp = 0.6
# Tl = 2


# # Calcular el espectro
# espectroSa = calcular_espectro_E030_PERU(
#     periodos_vibracion, Z, S, R, U, Tp, Tl)

# espectroSa
# # # Mostrar resultados
# # print("Espectro de diseño para los periodos de vibración:")
# # for tn, value in zip(Tn, espectro):
# #     print(f"Periodo: {tn} s, Espectro: {value:.2f}")


# Periodos_plot = np.arange(0, 12, 0.1)
# espectros = calcular_espectro_E030_PERU(Periodos_plot, Z, S, R, U, Tp, Tl)/981

# plt.plot(Periodos_plot, espectros, linestyle='-', linewidth=2, color='r',
#          label='Espectro de Diseño E030 Peru')
# plt.xlabel('Periodo (s)')
# plt.ylabel('Aceleracion (a/g)')
# plt.title('Espectro de Diseño E030 Peru')
# plt.grid(True)
# plt.legend()
# plt.show()
# # espectros


# # espectro de diseño de desplazamiento
# espectroSd = espectroSa / (2 * np.pi / periodos_vibracion)**2
# espectroSd


# import numpy as np
# import pandas as pd

# # Calcular deformaciones
# deformaciones = []
# for i in range(autovectores.shape[1]):
#     deformacion = autovectores[:, i] * Li[i] * espectroSd[i] / Mi[i]
#     deformaciones.append(deformacion)
# deformaciones = np.array(deformaciones).T

# # Convertir deformaciones a DataFrame
# df_deformaciones = pd.DataFrame(deformaciones).round(2)

# # Calcular la suma absoluta de todos los elementos de cada fila
# suma_absoluta_filas = df_deformaciones.abs().sum(axis=1)

# # Calcular SRSS (Suma de Raíces Cuadradas)
# SRSS = np.sqrt((df_deformaciones ** 2).sum(axis=1))

# # Calcular CQC (Combinación Cuadrática Completa)


# def CQC(df):
#     n_modes = df.shape[1]
#     cqc = []
#     for i in range(df.shape[0]):
#         suma = 0
#         for j in range(n_modes):
#             for k in range(n_modes):
#                 suma += df.iloc[i, j] * df.iloc[i, k] * (1 if j == k else 0.5)
#         cqc.append(np.sqrt(suma))
#     return np.array(cqc)


# CQC_values = CQC(df_deformaciones)

# # Agregar las columnas de combinaciones y suma absoluta al DataFrame
# df_deformaciones['Suma Absoluta'] = suma_absoluta_filas.round(2)
# df_deformaciones['SRSS'] = SRSS.round(2)
# df_deformaciones['CQC'] = CQC_values.round(2)

# # Mostrar el DataFrame actualizado
# print("DataFrame de Deformaciones con Suma Absoluta, SRSS y CQC:")
# print(df_deformaciones)


# import numpy as np
# import pandas as pd

# # Calcular fuerzas de entrepiso FI = (M.Φ[i]).Li[i]*Sa[i]/Mi[i]
# fuerzas_entrepiso = []
# for i in range(autovectores.shape[1]):
#     fuerza_entrepiso = np.dot(
#         M, autovectores[:, i]) * Li[i] * espectroSa[i] / Mi[i]
#     fuerzas_entrepiso.append(fuerza_entrepiso)
# fuerzas_entrepiso = np.array(fuerzas_entrepiso).T

# # Convertir fuerzas cortantes a DataFrame
# df_fuerzas_cortantes = pd.DataFrame(fuerzas_entrepiso).round(2)

# # Calcular la suma absoluta de todos los elementos de cada fila
# suma_absoluta_filas = df_fuerzas_cortantes.abs().sum(axis=1)

# # Calcular SRSS (Suma de Raíces Cuadradas)
# SRSS = np.sqrt((df_fuerzas_cortantes ** 2).sum(axis=1))

# # Calcular CQC (Combinación Cuadrática Completa)


# def CQC(df):
#     n_modes = df.shape[1]
#     cqc = []
#     for i in range(df.shape[0]):
#         suma = 0
#         for j in range(n_modes):
#             for k in range(n_modes):
#                 suma += df.iloc[i, j] * df.iloc[i, k] * (1 if j == k else 0.5)
#         cqc.append(np.sqrt(suma))
#     return np.array(cqc)


# CQC_values = CQC(df_fuerzas_cortantes)

# # Agregar las columnas de combinaciones y suma absoluta al DataFrame
# df_fuerzas_cortantes['Suma Absoluta'] = suma_absoluta_filas.round(2)
# df_fuerzas_cortantes['SRSS'] = SRSS.round(2)
# df_fuerzas_cortantes['CQC'] = CQC_values.round(2)

# # Mostrar el DataFrame actualizado
# print("DataFrame de Fuerzas de entrepiso con Suma Absoluta, SRSS y CQC:")
# print(df_fuerzas_cortantes)


# import numpy as np
# import pandas as pd


# def calcular_fuerzas_cortantes_entrepiso(fuerzas_entrepiso):
#     # Convertir la lista de listas a una matriz de numpy
#     matriz = np.array(fuerzas_entrepiso)

#     # Obtener el número de filas y columnas
#     filas, columnas = matriz.shape

#     # Crear una matriz de ceros del mismo tamaño para el resultado
#     fuerzas_cortantes_entrepiso = np.zeros((filas, columnas))

#     # Iterar sobre cada columna
#     for col in range(columnas):
#         suma_acumulativa = 0
#         for row in range(filas-1, -1, -1):
#             if matriz[row, col] != 0:
#                 suma_acumulativa += matriz[row, col]
#                 fuerzas_cortantes_entrepiso[row, col] = suma_acumulativa

#     return fuerzas_cortantes_entrepiso


# # Calcular fuerzas cortantes de entrepiso
# fuerzas_cortantes_entrepiso = calcular_fuerzas_cortantes_entrepiso(
#     fuerzas_entrepiso)

# # Convertir fuerzas cortantes a DataFrame
# df_fuerzas_cortantes_entrepiso = pd.DataFrame(
#     fuerzas_cortantes_entrepiso).round(2)

# # Calcular la suma absoluta de todos los elementos de cada fila
# suma_absoluta_filas = df_fuerzas_cortantes_entrepiso.abs().sum(axis=1)

# # Calcular SRSS (Suma de Raíces Cuadradas)
# SRSS = np.sqrt((df_fuerzas_cortantes_entrepiso ** 2).sum(axis=1))

# # Calcular CQC (Combinación Cuadrática Completa)


# def CQC(df):
#     n_modes = df.shape[1]
#     cqc = []
#     for i in range(df.shape[0]):
#         suma = 0
#         for j in range(n_modes):
#             for k in range(n_modes):
#                 suma += df.iloc[i, j] * df.iloc[i, k] * (1 if j == k else 0.5)
#         cqc.append(np.sqrt(suma))
#     return np.array(cqc)


# CQC_values = CQC(df_fuerzas_cortantes_entrepiso)

# # Agregar las columnas de combinaciones y suma absoluta al DataFrame
# df_fuerzas_cortantes_entrepiso['Suma Absoluta'] = suma_absoluta_filas.round(2)
# df_fuerzas_cortantes_entrepiso['SRSS'] = SRSS.round(2)
# df_fuerzas_cortantes_entrepiso['CQC'] = CQC_values.round(2)

# # Mostrar el DataFrame actualizado
# print("DataFrame de Fuerzas Cortantes de Entrepiso con Suma Absoluta, SRSS y CQC:")
# print(df_fuerzas_cortantes_entrepiso)


# 3
############
# 33
########
# 3


import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def read_xlsx(file_name: str) -> np.array:
    df = pd.read_excel(file_name, header=None)
    return df.to_numpy()


# Leer las matrices K y M desde los archivos Excel
K = read_xlsx("K.xlsx")
M = read_xlsx("M.xlsx")

# Calcular autovalores y autovectores
autovalores, autovectores = eigh(K, M)

# Crear DataFrames para los autovalores y autovectores
df_autovalores = pd.DataFrame(autovalores, columns=['Autovalores'])
df_autovectores = pd.DataFrame(autovectores)

# Mostrar los DataFrames
print("DataFrame de Autovalores:")
print(df_autovalores)

print("\nDataFrame de Autovectores:")
print(df_autovectores)

# Normalización de los autovectores al primer piso
for i in range(autovectores.shape[1]):
    normalizador = next(j for j in autovectores[:, i] if j != 0)
    autovectores[:, i] /= normalizador

# Convertir a DataFrame para visualizar
df_autovectores_normalizados = pd.DataFrame(autovectores).round(2)
print("\nAutovectores normalizados al primer piso:")
print(df_autovectores_normalizados)

# Cálculo de frecuencias naturales y períodos de vibración
frecuencias_naturales = np.sqrt(autovalores)
periodos_vibracion = 2 * np.pi / frecuencias_naturales

# Valores proporcionados para el cálculo del espectro
Z = 0.25
S = 1.2
R = 8
U = 1
Tp = 0.6
Tl = 2


def calcular_espectro_E030_PERU(Tn, Z, S, R, U, Tp, Tl) -> np.array:
    espectro_base = (Z * U * S) / R
    espectro = []
    for tn in Tn:
        if tn < Tp:
            C = 2.5
        elif Tp <= tn <= Tl:
            C = 2.5 * (Tp / tn)
        else:
            C = 2.5 * ((Tp * Tl) / tn**2)
        espectro.append(espectro_base * C * 981)
    return np.array(espectro)


# Calcular el espectro para los períodos de vibración
espectroSa = calcular_espectro_E030_PERU(
    periodos_vibracion, Z, S, R, U, Tp, Tl)

# Graficar el espectro de diseño
Periodos_plot = np.arange(0, 12, 0.1)
espectros = calcular_espectro_E030_PERU(
    Periodos_plot, Z, S, R, U, Tp, Tl) / 981

plt.plot(Periodos_plot, espectros, linestyle='-', linewidth=2,
         color='r', label='Espectro de Diseño E030 Perú')
plt.xlabel('Periodo (s)')
plt.ylabel('Aceleración (a/g)')
plt.title('Espectro de Diseño E030 Perú')
plt.grid(True)
plt.legend()
plt.show()

# Cálculo de espectro de diseño de desplazamiento
espectroSd = espectroSa / (2 * np.pi / periodos_vibracion)**2

# Cálculo de deformaciones
deformaciones = []
for i in range(autovectores.shape[1]):
    deformacion = autovectores[:, i] * Li[i] * espectroSd[i] / Mi[i]
    deformaciones.append(deformacion)
deformaciones = np.array(deformaciones).T

# Convertir deformaciones a DataFrame
df_deformaciones = pd.DataFrame(deformaciones).round(2)

# Cálculo de la suma absoluta, SRSS y CQC
suma_absoluta_filas = df_deformaciones.abs().sum(axis=1)
SRSS = np.sqrt((df_deformaciones ** 2).sum(axis=1))


def CQC(df):
    n_modes = df.shape[1]
    cqc = []
    for i in range(df.shape[0]):
        suma = 0
        for j in range(n_modes):
            for k in range(n_modes):
                suma += df.iloc[i, j] * df.iloc[i, k] * (1 if j == k else 0.5)
        cqc.append(np.sqrt(suma))
    return np.array(cqc)


CQC_values = CQC(df_deformaciones)

# Agregar las columnas de combinaciones al DataFrame
df_deformaciones['Suma Absoluta'] = suma_absoluta_filas.round(2)
df_deformaciones['SRSS'] = SRSS.round(2)
df_deformaciones['CQC'] = CQC_values.round(2)

# Mostrar el DataFrame actualizado
print("\nDataFrame de Deformaciones con Suma Absoluta, SRSS y CQC:")
print(df_deformaciones)

# Cálculo de fuerzas de entrepiso
fuerzas_entrepiso = []
for i in range(autovectores.shape[1]):
    fuerza_entrepiso = np.dot(
        M, autovectores[:, i]) * Li[i] * espectroSa[i] / Mi[i]
    fuerzas_entrepiso.append(fuerza_entrepiso)
fuerzas_entrepiso = np.array(fuerzas_entrepiso).T

# Convertir fuerzas cortantes a DataFrame
df_fuerzas_cortantes = pd.DataFrame(fuerzas_entrepiso).round(2)

# Cálculo de la suma absoluta, SRSS y CQC para fuerzas de entrepiso
suma_absoluta_filas = df_fuerzas_cortantes.abs().sum(axis=1)
SRSS = np.sqrt((df_fuerzas_cortantes ** 2).sum(axis=1))

CQC_values = CQC(df_fuerzas_cortantes)

# Agregar las columnas de combinaciones al DataFrame
df_fuerzas_cortantes['Suma Absoluta'] = suma_absoluta_filas.round(2)
df_fuerzas_cortantes['SRSS'] = SRSS.round(2)
df_fuerzas_cortantes['CQC'] = CQC_values.round(2)

# Mostrar el DataFrame actualizado
print("\nDataFrame de Fuerzas de Entrepiso con Suma Absoluta, SRSS y CQC:")
print(df_fuerzas_cortantes)

# Cálculo de fuerzas cortantes de entrepiso


def calcular_fuerzas_cortantes_entrepiso(fuerzas_entrepiso):
    matriz = np.array(fuerzas_entrepiso)
    filas, columnas = matriz.shape
    fuerzas_cortantes_entrepiso = np.zeros((filas, columnas))
    for col in range(columnas):
        suma_acumulativa = 0
        for row in range(filas-1, -1, -1):
            if matriz[row, col] != 0:
                suma_acumulativa += matriz[row, col]
                fuerzas_cortantes_entrepiso[row, col] = suma_acumulativa
    return fuerzas_cortantes_entrepiso


# Calcular fuerzas cortantes de entrepiso
fuerzas_cortantes_entrepiso = calcular_fuerzas_cortantes_entrepiso(
    fuerzas_entrepiso)

# Convertir fuerzas cortantes a DataFrame
df_fuerzas_cortantes_entrepiso = pd.DataFrame(
    fuerzas_cortantes_entrepiso).round(2)

# Cálculo de la suma absoluta, SRSS y CQC para fuerzas cortantes de entrepiso
suma_absoluta_filas = df_fuerzas_cortantes_entrepiso.abs().sum(axis=1)
SRSS = np.sqrt((df_fuerzas_cortantes_entrepiso ** 2).sum(axis=1))

CQC_values = CQC(df_fuerzas_cortantes_entrepiso)

# Agregar las columnas de combinaciones al DataFrame
df_fuerzas_cortantes_entrepiso['Suma Absoluta'] = suma_absoluta_filas.round(2)
df_fuerzas_cortantes_entrepiso['SRSS'] = SRSS.round(2)
df_fuerzas_cortantes_entrepiso['CQC'] = CQC_values.round(2)

# Mostrar el DataFrame actualizado
print("\nDataFrame de Fuerzas Cortantes de Entrepiso con Suma Absoluta, SRSS y CQC:")
print(df_fuerzas_cortantes_entrepiso)
