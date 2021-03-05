import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
from scipy.stats import spearmanr

import constants

# --------- VARIABLES DE CONTROL ---------- #
#
# Polinomios de Chebyshev
# x0, y0, m ,n
CHEBYSHEV_X_0 = 0.4811295983708204
CHEBYSHEV_Y_0 = -0.3437604907376320
CHEBYSHEV_N = 5.0468764305551317
CHEBYSHEV_M = 9.0078127384258552

# Funcion Tienda
# a
TIENDA_U = 1.5546548465465465
TIENDA_X_0 = 0.5811295983708204

# Funcion Logistica
# u
LOGISTICA_U = 3.71
LOGISTICA_X_0 = 0.6734678425981357
# ---------- VARIABLES DE CONTROL ---------- #

# ----------- VARIABLES GLOBALES ----------- #
#
# Global num of bytes of data
total_bytes = 0
# ----------- VARIABLES GLOBALES ----------- #

tracks = [['lion', 'audio/lion-encrypted.wav'], ['AnnenMayKantereit-19', 'audio/AnnenMayKantereit-19-encrypted.wav'],
          ['Silhouettes-19', 'audio/Silhouettes-19-encrypted.wav']]


def show_info(aname, a):
    print("|------| HEADER INFO |------|")
    # print("Array:", aname)
    print("shape:", a.shape)
    print("dtype:", a.dtype)
    print("min, max:", a.min(), a.max())
    print("|-------------------------|")


def plot_info(data):
    time = np.linspace(0., data.shape[0], data.shape[0])
    plt.plot(time, data[:, 0], label="Left channel")
    # plt.plot(time, data[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


rate, data = scipy.io.wavfile.read(tracks[2][1])

show_info("data", data)

# print("rate:", rate)
# print("dataset samples:", data[22:54, :])
bytes_data = bytes(data[0])

data_header = data[0:22, :]
show_info("headerData", data_header)
print("dataset header:", data[0:22, :])

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
    data_right = data[22:, 1]

    data_left = data[22:, 0]

    # Initialize numpy array using left data(mono) as int16
    data_array = np.array(data_left, dtype=np.int16)

    print("Muestras de audio 16-bit con signo: \n", data_left[100:120])

    # Translate the range (-32768, 32767) to positive values
    data_array = data_array + 32768
    # print("translate array:", data_array[100:105])
    # print("16-bit trasladados: \n", data_array[100:120])

    # Function int value to string binary
    to_string_function = np.vectorize(lambda x: np.binary_repr(x).zfill(m))

    # Generate binary string array from int16 array
    strings_array = to_string_function(data_array)
    # print("string_array:", strings_array[0:5])
    # print("list -> data_array.shape:", list(data_array.shape))

    # Create new array binaries_array with [[16],[16], ... [16]] shape
    binaries_array = np.zeros(list(data_array.shape) + [16], dtype=np.int8)
    # print("zeros -> binaries_array", binaries_array)

    # Create new array with [[4],[4], ... [4]] shape of binaries_array*4 length
    bytes_array = np.zeros((data_array.shape[0]*4), dtype=np.int8)
    bytes_array = np.zeros(list(bytes_array.shape) + [4], dtype=np.int8)

    # Function to return bit to bit to a [16] vector
    vectorize_bits_function = np.vectorize(lambda x: x[i_byte])

    # Covert all array to bit to bit vectors
    for i_byte in range(0, m):
        binaries_array[..., i_byte] = vectorize_bits_function(
            strings_array).astype("int16")

    # print("Vectores 16-bit de audio obtenidos: \n", binaries_array[100:120])
    # print("Length binaries_array: ", binaries_array.shape[0])
    # print("Length bytes_array (16): ", bytes_array.shape[0])

    # Set global value num of bytes in data
    total_bytes = bytes_array.shape[0]

    # Generation of key blocks
    bytes_key_arrays = key_data(
        CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N, CHEBYSHEV_M, total_bytes)

    # print("Keys muestra: ", bytes_key_arrays[10:18])
    # print("Keys muestra: ", bytes_key_arrays[0:10])

    # Split bytes vectors
    j = 0
    for i in binaries_array:
        # Divide [16] vector into [4][4][4][4]
        bytes_array[j] = np.array_split(i, 4)[0]
        # print("bytes_array[",j,"]", bytes_array[j])
        j += 1
        bytes_array[j] = np.array_split(i, 4)[1]
        # print("bytes_array[",j,"]", bytes_array[j])
        j += 1
        bytes_array[j] = np.array_split(i, 4)[2]
        # print("bytes_array[",j,"]", bytes_array[j])
        j += 1
        bytes_array[j] = np.array_split(i, 4)[3]
        # print("bytes_array[",j,"]", bytes_array[j])
        if j == bytes_array.shape[0]-1:
            break
        else:
            j += 1

    # print("Vectores 4-bit de audio obtenidos: \n", bytes_array[100:120])
    print("Vectores 4-bit de audio obtenidos shape: \n", bytes_array.shape)
    total_bytes = bytes_array.shape[0]
    print("Cantidad de vectores de audio obtenidos: \n", total_bytes)

    # print("Lenght bytes array (4): ", bytes_array.shape[0])
    # print("Lenght bytes key array (4): ", bytes_key_arrays.shape[0])

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

    # print("Claves generadas: \n", list_keys[0:5])
    list_keys = vectorize_bits_array(list_keys, 4)
    # print("Claves de 4-bit obtenidas: \n", list_keys[0:10])

    return list_keys


def permutation_data(lenght):

    print("Numero de vectores en permutation_data: ", lenght)

    def tend_map_iterator(x_0):
        x_i = 0
        if x_0 >= 0 and x_0 < 0.5:
            x_i = x_0 * TIENDA_U
        else:
            x_i = (1 - x_0) * TIENDA_U

        return x_i

    def logistic_map_iterator(x_0):
        x_i = LOGISTICA_U * x_0 * (1 - x_0)
        return x_i

    list_positions_tent = []
    list_positions_logistic = []
    x_i = TIENDA_X_0
    y_i = LOGISTICA_X_0

    for i in range(lenght + 51):
        x_i = tend_map_iterator(x_i)
        if i > 50:
            x_final = math.floor((x_i * 10**2) % 4)
            # POR REVISAR, ELECCION DE NUMERO
            list_positions_tent.append(x_final)
            # print("tent value: ", x_final)
            # print("tent value in list: ", list_positions_tent[i])

    longitud_bloque = 625
    cantidad_bloques = math.floor(lenght/longitud_bloque)
    print("Cantidad de bloques: ", cantidad_bloques)
    longitud_cola = lenght % longitud_bloque
    print("Longitud cola: ", longitud_cola)
    for i in range(cantidad_bloques + 52):
        y_i = logistic_map_iterator(y_i)
        if i > 50 and i < (cantidad_bloques + 51):
            y_final = math.floor((y_i * 10**2) % 4)
            list_positions_logistic.append(y_final)
        elif i > 50 and longitud_cola > 0:
            y_final = math.floor((y_i * 10**2) % longitud_cola)
            list_positions_logistic.append(y_final)

    # list_positions = vectorize_bits_array(list_positions, 8)
    # print("Tent positions: \n", list_positions_tent)
    print("Tent positions lenght: \n", len(list_positions_tent))
    # print("Logistic positions: \n", list_positions_logistic)
    print("Logistic positions lenght: \n", len(list_positions_logistic))
    return list_positions_tent, list_positions_logistic, longitud_cola, longitud_bloque


def permutate_data(data_array, position_data_bits, position_data_bytes, tail_lenght, block_lenght):

    slice_index = len(data_array)-tail_lenght
    array_head_reshaped = np.reshape(
        np.array(data_array[:slice_index]), (len(position_data_bytes) - 1, block_lenght, 4))
    array_tail_reshaped = np.reshape(
        np.array(data_array[slice_index:]), (1, tail_lenght, 4))

    permuted_data_bytes_head = np.zeros(
        list(array_head_reshaped.shape), dtype=np.int8)
    for i in range(len(position_data_bytes) - 1):
        permuted_data_bytes_head[i] = np.roll(array_head_reshaped[i],
                                              position_data_bytes[i] * -1, axis=0)

    # print(i)
    # print("permuted_data_bytes_head shape: \n", permuted_data_bytes_head.shape)

    permuted_data_bytes_2d_head = permuted_data_bytes_head.reshape(
        (len(position_data_bits) - tail_lenght), 4)
    # print("permuted_data_bytes_head 2d: \n", permuted_data_bytes_2d_head)

    permuted_data_bytes_tail = np.roll(
        array_tail_reshaped[0], position_data_bytes[len(position_data_bytes) - 1] * -1, axis=0)
    # print("permuted_data_bytes_tail shape: \n", permuted_data_bytes_tail.shape)
    # print("permuted_data_bytes_tail 2d: \n", permuted_data_bytes_tail)

    permuted_data = np.concatenate(
        (permuted_data_bytes_2d_head, permuted_data_bytes_tail), axis=0)
    # print("permuted_data: \n", permuted_data)
    print("permuted_data shape: \n", permuted_data.shape)

    permuted_data_bits = np.zeros(list(data_array.shape), dtype=np.int8)
    # print("Permuted data bits zeros \n", permuted_data_bits)

    # print("Data array \n", data_array)
    # print("Permuted data bits \n", permuted_data_bits)
    # print("Permuted data bits shape \n", permuted_data_bits.shape)

    for i in range(len(permuted_data)):
        # print("Posicion: ", position_data_bits[i])
        # print("Antes: ", data_array[i])
        # print("Despues: ", np.roll(data_array[i], position_data_bits[i]))
        permuted_data_bits[i] = np.roll(
            permuted_data[i], position_data_bits[i] * -1)

    return permuted_data_bits


def binary_array_to_int(array):
    result = 0
    for digits in array:
        result = (result << 1) | digits

    return result


def int_to_binary_array(num, m):
    # Convert a positive integer num into an m-bit bit vector
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def substitute_data(data_array):

    substituted_data = []

    j = 0
    for byte in data_array:
        if j < 10:
            # print("byte: ", byte)
            num = binary_array_to_int(byte)
            # print("num: ", num)
            # print("inverse: ", pow(num.item(), 15, 17))
            inverse = pow(num.item(), 15, 17)
            # print("binary: ", int_to_binary_array(inverse, 8))
            substituted_data.append(int_to_binary_array(inverse, 4))
        else:
            # print("byte: ", byte)
            num = binary_array_to_int(byte)
            # print("num: ", num)
            # print("inverse: ", pow(num.item(), 15, 17))
            inverse = pow(num.item(), 15, 17)
            # print("binary: ", int_to_binary_array(inverse, 8))
            substituted_data.append(int_to_binary_array(inverse, 4))
        j += 1

    return substituted_data


def convert_to_wav_file(data_array):
    binaries_array = np.zeros(int(len(data_array) / 4), dtype=np.int16)
    # binaries_array = np.zeros(
    #     list(binaries_array.shape) + [16], dtype=np.int16)

    j = 0
    for i in range(0, len(binaries_array)):
        binaries_array[i] = binary_array_to_int(
            np.concatenate((data_array[j], data_array[j+1], data_array[j+2], data_array[j+3]), axis=0))
        # if j < 10:
        #     print("Type: ", type(data_array[j]))
        #     print("Concatenated 1: ", data_array[j])
        #     print("Concatenated 2: ", data_array[j+1])
        #     print("Concatenated: ", np.concatenate((data_array[j], data_array[j+1])))
        #     print("---------->: ", binaries_array[i])

        j += 4
        if j >= len(data_array):
            break

    # print("Frecuencias trasladadas: ", binaries_array[0:10])

    binaries_array = binaries_array - 32768

    # print("Frecuencias: ", binaries_array[0:10])
    # print("Cantidad frecuencias: ", len(binaries_array))

    final_array = np.empty_like(data)
    # print("Final data: ", final_array[0:43])
    show_info("binariesArray", binaries_array)
    final_array[:, 0] = np.concatenate((data_header[:, 0], binaries_array))
    final_array[:, 1] = np.concatenate((data_header[:, 1], binaries_array))
    final_array = final_array.astype(np.int16)
    # print("Final data: ", final_array[22:54])

    # plot_info(data)
    # plot_info(final_array)

    scipy.io.wavfile.write(
        'audio/' + tracks[2][0] + '-decrypted.wav', rate, final_array)

    # new_rate, new_data = scipy.io.wavfile.read(
    #     'audio/lion-decrypted.wav')
    # show_info("NewData", new_data)

    corr, _ = pearsonr(data[:, 0], final_array[:, 0])
    print('Pearsons correlation: %.3f' % corr)

    corr, _ = spearmanr(data[:, 0], final_array[:, 0])
    print('Spearmans correlation: %.3f' % corr)


def get_statics(init_data, final_data):
    corr, _ = pearsonr(data[:, 0], final_array[:, 0])
    print('Pearsons correlation: %.3f' % corr)

    corr, _ = spearmanr(data[:, 0], final_array[:, 0])
    print('Spearmans correlation: %.3f' % corr)


data_array, key_array = stack_data(data, 16)
# print('-----------------------------------------')
# print('-----------------------------------------')
# print('-----------------------------------------')
# print("Cantidad total de bytes de data: ", len(data_array))
# print("Muestra matrices data:")
# print(data_array[10:18])

substituted_array = substitute_data(data_array)
# print("Muestra substituted data:")
# print(substituted_array[10:18])

xor_array = np.bitwise_xor(substituted_array, key_array)
# print("Cantidad bytes de clave: ", len(key_array))
# print("Muestra clave data:")
# print(key_array[10:18])
# print("Muestra xor data:")
# print(xor_array[10:18])

# positions_array = permutation_data(len(xor_array))

positions_bits, positions_bytes, tail_lenght, block_lenght = permutation_data(
    len(xor_array))
# print("Cantidad positions: ", len(positions_array))
# print("Muestra positions data:")
# print(positions_array[10:18])

# permuted_array = permutate_data(xor_array, positions_array)

permuted_array = permutate_data(
    xor_array, positions_bits, positions_bytes, tail_lenght, block_lenght)
# print("Muestra permuted data:")
# print(permuted_array[10:18])

print("|---------------------------------------------------------|")
print("|---------------------------------------------------------|")

convert_to_wav_file(permuted_array)
