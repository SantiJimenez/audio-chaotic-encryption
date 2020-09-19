import scipy.io.wavfile
import numpy as np
import math

import constants

# --------- VARIABLES DE CONTROL ---------- #
#
# Polinomios de Chebyshev
# x0, y0, m ,n
CHEBYSHEV_X_0 = 0.48112959837082048697
CHEBYSHEV_Y_0 = -0.3437604907376320
CHEBYSHEV_N = 5.0468764305551317
CHEBYSHEV_M = 23.0078127384258552

# Funcion Tienda
# a
TIENDA_A = 0.0525434533533456
# ---------- VARIABLES DE CONTROL ---------- #

# ----------- VARIABLES GLOBALES ----------- #
#
# Global num of bytes of data
total_bytes = 0
# ----------- VARIABLES GLOBALES ----------- #


def show_info(aname, a):
    print("|------| DATA INFO |------|")
    print("Array:", aname)
    print("shape:", a.shape)
    print("dtype:", a.dtype)
    print("min, max:", a.min(), a.max())
    print("|-------------------------|")


rate, data = scipy.io.wavfile.read('lion.wav')

show_info("data", data)
print("rate:", rate)
print("dataset:", data[100:110, :])
bytes_data = bytes(data[0])

# def int_to_string(x, m):
#     return np.vectorize(np.binary_repr(x).zfill(m))


def vectorize_bits_array(array, m):
    int_to_string_function = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strings_array = int_to_string_function(array)

    vectorize_bits_function = np.vectorize(lambda x: x[i_byte])

    bytes_array = np.zeros((len(array)), dtype=np.int8)
    bytes_array = np.zeros(list(bytes_array.shape) + [m], dtype=np.int8)

    for i_byte in range(0, m):
        bytes_array[..., i_byte] = vectorize_bits_function(
            strings_array).astype("int8")

    # print("Bloques de claves: ", bytes_array[0:10])
    return bytes_array


def stack_data(data, m):
    data_right = data[:, 1]

    data_left = data[:, 0]

    # Initialize numpy array using left data(mono) as int16
    data_array = np.array(data_left, dtype=np.int16)

    # Translate the range (-32768, 32769) to positive values
    data_array = data_array + 32768
    # print("translate array:", data_array[100:105])

    # Function int value to string binary
    to_string_function = np.vectorize(lambda x: np.binary_repr(x).zfill(m))

    # Generate binary string array from int16 array
    strings_array = to_string_function(data_array)
    print("string_array:", strings_array[100:105])
    # print("list -> data_array.shape:", list(data_array.shape))

    # Create new array binaries_array with [[16],[16], ... [16]] shape
    binaries_array = np.zeros(list(data_array.shape) + [16], dtype=np.int8)
    # print("zeros -> binaries_array", binaries_array)

    # Create new array with [[8],[8], ... [8]] shape of binaries_array*2 length
    bytes_array = np.zeros((data_array.shape[0]*2), dtype=np.int8)
    bytes_array = np.zeros(list(bytes_array.shape) + [8], dtype=np.int8)

    # Function to return bit to bit to a [16] vector
    vectorize_bits_function = np.vectorize(lambda x: x[i_byte])

    # Covert all array to bit to bit vectors
    for i_byte in range(0, m):
        binaries_array[..., i_byte] = vectorize_bits_function(
            strings_array).astype("int16")

    print("binaries_array: ", binaries_array[0:5])
    print("Length binaries_array: ", binaries_array.shape[0])
    print("Length bytes_array (16): ", bytes_array.shape[0])

    # Set global value num of bytes in data
    total_bytes = bytes_array.shape[0]

    # Generation of key blocks
    bytes_key_arrays = key_data(
        CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N, CHEBYSHEV_M, total_bytes)

    print("Keys muestra: ", bytes_key_arrays[0:10])

    # Split bytes vectors
    j = 0
    for i in binaries_array:
        # print(i)
        bytes_array[j] = np.array_split(i, 2)[0]
        # print("bytes_array[",j,"]", bytes_array[j])
        j += 1
        bytes_array[j] = np.array_split(i, 2)[1]
        # print("bytes_array[",j,"]", bytes_array[j])
        if j == bytes_array.shape[0]-1:
            break
        else:
            j += 1

    print("Bytes array muestra: ", bytes_array[0:10])
    print("Lenght bytes array (8): ", bytes_array.shape[0])
    print("Lenght bytes key array (8): ", bytes_key_arrays.shape[0])

    # Flatten data vector, delete vector structures
    flatten_array = bytes_array.flatten()
    flatten_array_keys = bytes_key_arrays.flatten()
    print("Flatten array muestra: ", flatten_array[0:24])
    print("Flatten array keys muestra: ", flatten_array_keys[0:24])

    # Creation of data matrix and tail
    cantidad_binarios_fila = 25
    cantidad_binarios_columna = 25
    cantidad_elementos_totales_matriz = cantidad_binarios_fila * \
        cantidad_binarios_columna  # 400 * 400
    num_matrices = len(flatten_array) // cantidad_elementos_totales_matriz

    matrices_datos_crudos = []
    matrices_keys_crudos = []
    el = 0
    print("Numero de matrices {}".format(num_matrices))
    for i in range(0, num_matrices):
        matriz_data = []
        matriz_key = []
        for r in range(0, cantidad_binarios_fila):
            row_data = []
            row_key = []
            for c in range(0, cantidad_binarios_columna):
                row_data.append(flatten_array[el])
                row_key.append(flatten_array_keys[el])
                el += 1
            matriz_data.append(row_data)
            matriz_key.append(row_key)
        matrices_datos_crudos.append(matriz_data)
        matrices_keys_crudos.append(matriz_key)
        # print("size matrices w ={}x{}".format(len(matriz_data[0]), len(matriz_data)))

    cola = []
    for j in range(el, len(flatten_array)):
        cola.append(flatten_array[j])
    print("Len cola = {}".format(len(cola)))

    # print("matrices muestras: ", matrices_datos_crudos[0][10:11])
    # print("matrices keys muestras: ", matrices_keys_crudos[0][10:11])

    # return bytes_array
    return matrices_datos_crudos, cola, matrices_keys_crudos


def key_data(x_0, y_0, n, m, lenght):

    def polynomialSequenceX(x): return math.cos(n*math.acos(x))
    def polynomialSequenceY(y): return math.cos(m*math.acos(y))

    def normalizationValues(x):
        n1 = math.floor(x * 10**5)
        n2 = math.floor(x * 10**9) - (n1 * 10**4)
        return (n1 ^ n2) % 256

    x = x_0
    y = y_0

    list_keys = []
    for i in range(0, int(lenght/2)):
        x = polynomialSequenceX(x)
        y = polynomialSequenceY(y)

        list_keys.append(normalizationValues(
            x))
        list_keys.append(normalizationValues(
            y))

    list_keys = vectorize_bits_array(list_keys, 8)

    # print("List keys: ", list_keys)

    return list_keys


def new_position_data(x_0):


data_matrix, tail, key_matrix = stack_data(data, 16)
print("Cantidad matrices: ", len(data_matrix))
print("Logitud cola: ", len(tail))

xor_data = np.bitwise_xor(data_matrix, key_matrix)

print("Muestra matrices data: ", data_matrix[0][10][0:8])
print("Muestra keys data:     ", key_matrix[0][10][0:8])
print("Muestra xor data:      ", xor_data[0][10][0:8])

# key_list = key_data(CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N, CHEBYSHEV_M)
# print("Muestra matrices claves: ", key_list)
