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
