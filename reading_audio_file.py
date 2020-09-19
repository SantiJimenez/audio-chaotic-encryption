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
TIENDA_X_0 = 0.48112959837082048697
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
    # print("Length bytes_array (16): ", bytes_array.shape[0])

    # Set global value num of bytes in data
    total_bytes = bytes_array.shape[0]

    # Generation of key blocks
    bytes_key_arrays = key_data(
        CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N, CHEBYSHEV_M, total_bytes)

    # print("Keys muestra: ", bytes_key_arrays[0:10])

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

    # print("matrices muestras: ", matrices_datos_crudos[0][10:11])
    # print("matrices keys muestras: ", matrices_keys_crudos[0][10:11])

    # return bytes_array
    return bytes_array, bytes_key_arrays


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


def permutation_data(lenght):

    def tend_map_iterator(x_0):
        x_i = 0
        if x_0 >= 0 and x_0 <= TIENDA_A:
            x_i = x_0/TIENDA_A
        elif x_0 > TIENDA_A and x_0 <= 1:
            x_i = (1 - x_0)/(1 - TIENDA_A)

        return x_i

    list_positions = []
    x_i = TIENDA_X_0
    for i in range(lenght + 51):
        x_i = tend_map_iterator(x_i)
        if i > 50:
            x_final = math.floor((x_i * 10**2) % 8)
            # POR REVISAR, ELECCION DE NUMERO
            list_positions.append(x_final)

    # list_positions = vectorize_bits_array(list_positions, 8)
    return list_positions


def permutate_data(data_array, position_data):
    permuted_data = []
    for i in range(len(data_array)):
        if i < 10:
            print("vector original:  ",
                  data_array[i], " -> ", position_data[i])
            print("vector permutado: ", np.roll(
                data_array[i], position_data[i]))
        permuted_data.append(np.roll(data_array[i], position_data[i]))

    return permuted_data


def binary_array_to_int(array):
    result = 0
    for digits in array:
        result = (result << 1) | digits

    return result


def int_to_binary_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def substitute_data(data_array):

    substituted_data = []

    for byte in data_array:
        # print("byte: ", byte)
        num = binary_array_to_int(byte)
        # print("num: ", num)
        # print("inverse: ", pow(num.item(), 15, 17))
        inverse = pow(num.item(), 15, 17)
        # print("binary: ", int_to_binary_array(inverse, 8))
        substituted_data.append(int_to_binary_array(inverse, 8))

    return substituted_data


data_array, key_array = stack_data(data, 16)
print("Cantidad bytes: ", len(data_array))
print("Muestra matrices data:")
print(data_array[10:18])

positions_array = permutation_data(len(data_array))
print("Cantidad positions: ", len(positions_array))
print("Muestra positions data:")
print(positions_array[10:18])

permuted_array = permutate_data(data_array, positions_array)
print("Muestra permuted data:")
print(permuted_array[10:18])

xor_array = np.bitwise_xor(permuted_array, key_array)
print("Cantidad bytes de clave: ", len(key_array))
print("Muestra clave data:")
print(key_array[10:18])
print("Muestra xor data:")
print(xor_array[10:18])

substituted_array = substitute_data(xor_array)
print("Muestra substituted data:")
print(substituted_array[10:18])

print("|---------------------------------------------------------|")
print("|---------------------------------------------------------|")

# key_list = key_data(CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N, CHEBYSHEV_M)
# print("Muestra matrices claves: ", key_list)
