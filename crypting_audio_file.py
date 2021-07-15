from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy
import scipy.io.wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import constants

# --------- VARIABLES DE CONTROL ---------- #
#
# Polinomios de Chebyshev
# x0, y0, m ,n
CHEBYSHEV_X_0 = 0.4811295983708204
CHEBYSHEV_Y_0 = -0.8437604907376320
CHEBYSHEV_N = 5.0468764305551317
CHEBYSHEV_M = 1.0078127384258552

# Funcion Tienda
# a
TIENDA_U = 1.5546548465465465
TIENDA_X_0 = 0.5811295983708204

# Funcion Logistica
# u
LOGISTICA_U = 3.8778654516456545
LOGISTICA_X_0 = 0.6734678425981357
# ---------- VARIABLES DE CONTROL ---------- #

# ----------- VARIABLES GLOBALES ----------- #
#
# Global num of bytes of data
total_bytes = 0
# ----------- VARIABLES GLOBALES ----------- #

tracks = [['audio/Lion-1.wav', 'Lion-1'], ['audio/Annen-2.wav', 'Annen-2'], ['audio/Editors-4.wav', 'Editors-4'],
          ['audio/SundaraK-10.wav', 'SundaraK-10'], ['audio/Aurora-18-2.wav', 'Aurora-18'], ['audio/Sam-30.wav', 'Sam-30']]


def generateSeeds():
    # np.random.seed(42)
    np.set_printoptions(precision=16)
    c_m = np.random.uniform(2, 20)
    c_n = np.random.uniform(2, 20)
    c_x = np.random.uniform(-0.8, 0.8)
    c_y = np.random.uniform(-0.8, 0.8)

    # t_u = np.random.uniform(0, 1)
    t_u = np.random.uniform(1, 1.7)
    t_x = np.random.uniform(0, 0.7)

    l_u = 3.5778654516456545
    l_x = np.random.uniform(0, 0.7)
    # l_u = 3.7115645816513156
    # l_x = 0.6734678425981357

    print("     CX:", c_x)
    print("     CY:", c_y)
    print("     CM:", c_m)
    print("     CN:", c_n)
    print("     TU", t_u)
    print("     TX", t_x)
    print("     LU", l_u)
    print("     LX", l_x)

    return c_x, c_y, c_n, c_m, t_u, t_x, l_u, l_x


def show_info(aname, a):
    print("|------| ", aname, "INFO |------|")
    # print("Array:", aname)
    print("shape:", a.shape)
    print("dtype:", a.dtype)
    print("min, max:", a.min(), a.max())
    print("|-------------------------|")


def int_to_byte_array_funct(num):
    # Convert a positive integer num into an m-bit bit vector
    return np.array(list(np.binary_repr(num).zfill(8))).astype(np.int8)


def vectorize_bits_array(array, m):
    int_to_byte_array = np.array(list(map(int_to_byte_array_funct, array)))
    return np.reshape(int_to_byte_array, (len(int_to_byte_array)*2, m))


@profile
def stack_data(data, m, c_x, c_y, c_n, c_m):
    data_right = data[22:, 1]

    data_left = data[22:, 0]

    # Initialize numpy array using left data(mono) as int16
    data_array = np.array(data_left, dtype=np.int16)

    # print("Muestras de audio 16-bit con signo: \n", data_left[100:120])

    # Translate the range (-32768, 32767) to positive values
    data_array = data_array + 32768
    # print("Muestras de audio 16-bit trasladados: \n",
    #       data_array[100:120])

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
    # bytes_key_arrays = key_data(
    #     control_parameters[0], control_parameters[1], control_parameters[2], control_parameters[3], total_bytes)

    # bytes_key_arrays = key_data(
    #     CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N, CHEBYSHEV_M, total_bytes)

    bytes_key_arrays = key_data(
        c_x, c_y, c_n, c_m, total_bytes)

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

    # print("Vectores 4-bit de audio obtenidos: \n", bytes_array[400:420])
    # print("Vectores 4-bit de audio obtenidos shape: \n", bytes_array.shape)
    total_bytes = bytes_array.shape[0]
    # print("Cantidad de vectores de audio obtenidos: \n", total_bytes)

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

        norm_x = normalizationValues(x)
        norm_y = normalizationValues(y)

        final_c = np.bitwise_xor(norm_x, norm_y)
        # if i > 50:
        #     condition_x = normalizationValues(x)
        #     if condition_x % 2 == 0:
        #         list_keys.append(normalizationValues(
        #             x))
        #         list_keys.append(normalizationValues(
        #             y))
        #     else:
        #         list_keys.append(normalizationValues(
        #             y))
        #         list_keys.append(normalizationValues(
        #             x))
        # list_keys.append(normalizationValues(x))
        # list_keys.append(normalizationValues(y))
        list_keys.append(final_c % 256)

    # print("Longitud audio: \n", lenght)
    # print("Longitud claves: \n", len(list_keys))
    # print("Claves generadas: \n", list_keys[0:8])
    list_keys_bits = vectorize_bits_array(list_keys, 4)
    # print("Longitud claves: \n", len(list_keys_bits))
    # print("Claves de 4-bit obtenidas: \n", list_keys_bits[0:16])

    return list_keys_bits


def permutation_data(lenght, t_u, t_x, l_u, l_x):

    # print("Numero de vectores en permutation_data: ", lenght)

    def tend_map_iterator(x_0):
        x_i = 0
        if x_0 >= 0 and x_0 < 0.5:
            x_i = x_0 * t_u
        else:
            x_i = (1 - x_0) * t_u

        return x_i

    def logistic_map_iterator(x_0):
        x_i = l_u * x_0 * (1 - x_0)
        return x_i

    list_positions_tent = []
    list_positions_logistic = []
    x_i = t_x
    y_i = l_x

    for i in range(lenght + 51):
        x_i = tend_map_iterator(x_i)
        y_i = logistic_map_iterator(y_i)
        if i > 50:
            x_final = math.floor((x_i * 10**2) % 4)
            y_final = math.floor((y_i * 10**2) % 4)
            final_t = np.bitwise_xor(x_final, y_final)
            # POR REVISAR, ELECCION DE NUMERO
            list_positions_tent.append(final_t % 4)
        #     # print("tent value: ", x_final)
        #     # print("tent value in list: ", list_positions_tent[i])
        # x_i = tend_map_iterator(x_i)
        # if i > 50:
        #     x_final = math.floor((x_i * 10**2) % 4)
        #     # POR REVISAR, ELECCION DE NUMERO
        #     list_positions_tent.append(x_final)

    longitud_bloque = 255
    cantidad_bloques = math.floor(lenght/longitud_bloque)
    # print("Cantidad de bloques (+1 Cola): ", cantidad_bloques + 1)
    longitud_cola = lenght % longitud_bloque
    # print("Longitud cola: ", longitud_cola)
    for i in range(cantidad_bloques + 52):
        x_i = tend_map_iterator(x_i)
        y_i = logistic_map_iterator(y_i)
        if i > 50 and i < (cantidad_bloques + 51):
            x_final = math.floor((x_i * 10**3) % longitud_bloque)
            y_final = math.floor((y_i * 10**3) % longitud_bloque)
            final_l = np.bitwise_xor(x_final, y_final)
            list_positions_logistic.append(final_l % longitud_bloque)
        elif i > 50 and longitud_cola > 0:
            x_final = math.floor((x_i * 10**3) % longitud_cola)
            y_final = math.floor((y_i * 10**3) % longitud_cola)
            final_l = np.bitwise_xor(x_final, y_final)
            list_positions_logistic.append(final_l % longitud_cola)
            # y_final = math.floor((final_l * 10**3) % longitud_cola)
            # list_positions_logistic.append(y_final)

        # y_i = logistic_map_iterator(y_i)
        # if i > 50 and i < (cantidad_bloques + 51):
        #     y_final = math.floor((y_i * 10**3) % longitud_bloque)
        #     list_positions_logistic.append(y_final)
        # elif i > 50 and longitud_cola > 0:
        #     y_final = math.floor((y_i * 10**3) % longitud_cola)
        #     list_positions_logistic.append(y_final)

    # list_positions = vectorize_bits_array(list_positions, 8)
    # print("Tent positions: \n", list_positions_tent)
    # print("Tent positions lenght: \n", len(list_positions_tent))
    # print("Logistic positions: \n", list_positions_logistic)
    # print("Logistic positions lenght: \n", len(list_positions_logistic))
    return list_positions_tent, list_positions_logistic, longitud_cola, longitud_bloque


def permutate_data(data_array, position_data_bits, position_data_bytes, tail_lenght, block_lenght):

    permuted_data_bits = np.zeros(list(data_array.shape), dtype=np.int8)
    # print("ROTACIÓN DE BITS")
    for i in range(len(data_array)):
        # if i < 10:
        #     print("Muestra de audio ", i+1, ": \n", data_array[i], " --> ", np.roll(
        #         data_array[i], position_data_bits[i]))
        permuted_data_bits[i] = np.roll(data_array[i], position_data_bits[i])

    # print("Data array \n", data_array)
    # print("Permuted data bits \n", permuted_data_bits)
    # print("Permuted data bits shape \n", permuted_data_bits.shape)

    slice_index = len(data_array)-tail_lenght
    array_head_reshaped = np.reshape(
        np.array(permuted_data_bits[:slice_index]), (len(position_data_bytes) - 1, block_lenght, 4))
    array_tail_reshaped = np.reshape(
        np.array(permuted_data_bits[slice_index:]), (1, tail_lenght, 4))

    permuted_data_bytes_head = np.zeros(
        list(array_head_reshaped.shape), dtype=np.int8)
    # print("ROTACIÓN DE BLOQUES")
    for i in range(len(position_data_bytes) - 1):
        # if i < 2:
        #     print("Muestra de bloque de audio original: \n",
        #           array_head_reshaped[i])
        #     print("Muestra de bloque de audio rotada: \n", np.roll(
        #         array_head_reshaped[i], position_data_bytes[i], axis=0))
        permuted_data_bytes_head[i] = np.roll(array_head_reshaped[i],
                                              position_data_bytes[i], axis=0)

    # print(i)
    # print("permuted_data_bytes_head shape: \n", permuted_data_bytes_head.shape)

    permuted_data_bytes_tail = np.roll(
        array_tail_reshaped[0], position_data_bytes[len(position_data_bytes) - 1], axis=0)
    # print("permuted_data_bytes_tail shape: \n", permuted_data_bytes_tail.shape)
    # print("permuted_data_bytes_tail 2d: \n", permuted_data_bytes_tail)

    permuted_data_bytes_2d_head = permuted_data_bytes_head.reshape(
        (len(position_data_bits) - tail_lenght), 4)
    # print("permuted_data_bytes_head 2d: \n", permuted_data_bytes_2d_head)

    permuted_data = np.concatenate(
        (permuted_data_bytes_2d_head, permuted_data_bytes_tail), axis=0)
    # print("permuted_data: \n", permuted_data)
    # print("permuted_data shape: \n", permuted_data.shape)
    # print("Longitud data permutada:", len(permuted_data))
    return permuted_data
    # return permuted_data_bits


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
    # substituted_data = np.zeros(list(data_array.shape), dtype=np.int8)

    j = 0
    # print("- BITS ----- INT -- INVERSO ---- BITS")
    # for i in range(0, len(data_array)):
    for byte in data_array:
        num = binary_array_to_int(byte)
        # num = binary_array_to_int(data_array[i])
        inverse = pow(num.item(), 15, 17)
        substituted_data.append(int_to_binary_array(inverse, 4))
        # substituted_data[i] = int_to_binary_array(inverse, 4)
        # if j < 20:
        #     # print("[0 1 0 0] --> 10 ----> 15 ---> [0 1 0 0]")
        #     if num < 10 and inverse < 10:
        #         print(byte, "-->", num, " ---->", inverse,
        #               " --->", int_to_binary_array(inverse, 4))
        #     if num < 10 and inverse >= 10:
        #         print(byte, "-->", num, " ---->", inverse,
        #               "--->", int_to_binary_array(inverse, 4))
        #     if num >= 10 and inverse < 10:
        #         print(byte, "-->", num, "---->", inverse,
        #               " --->", int_to_binary_array(inverse, 4))
        #     if num >= 10 and inverse >= 10:
        #         print(byte, "-->", num, "---->", inverse,
        #               "--->", int_to_binary_array(inverse, 4))
        j += 1

    return substituted_data


def convert_to_wav_file_format(data_origin, data_array, data_header):
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

    # get_statics(data[22:, 1] binaries_array)

    binaries_array = binaries_array - 32768

    # print("Muestras de audio 16-bit cifradas: \n", binaries_array[0:20])
    # print("Cantidad frecuencias: ", len(binaries_array))

    final_array = np.empty_like(data_origin)
    # print("Final data: ", final_array[0:43])
    # show_info("DATA ARRAY", data_array)
    # show_info("BINARIES ARRAY", binaries_array)
    # show_info("FINAL ARRAY", binaries_array)
    final_array[:, 0] = np.concatenate((data_header[:, 0], binaries_array))
    final_array[:, 1] = np.concatenate((data_header[:, 1], binaries_array))
    final_array = final_array.astype(np.int16)
    # print("Final data: ", final_array[22:54])
    return final_array


def plots(data, title, rate):

    def waveform():
        time = np.linspace(0., data.shape[0], data.shape[0])

        plt.plot(time, data)
        # plt.plot(time, data[:, 0], label="Left channel")
        # plt.plot(time, data[:, 1], label="Right channel")
        # plt.legend()
        plt.title("Audio Waveform " + title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [dB]")
        plt.show()

    def histogram():
        plt.hist(data, bins='auto')  # arguments are passed to np.histogram.
        plt.title("Audio Histogram " + title)
        plt.xlabel("Amplitude [dB]")
        plt.ylabel("Number of samples")
        plt.show()

    def spectogram():
        plt.specgram(data, Fs=rate)
        plt.title("Audio Spectrogram " + title)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.show()

    waveform()
    histogram()
    spectogram()


def SNR(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def PSNR(original, encrypted, mse):
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_sample = 65535
    psnr = 10 * math.log10((max_sample**2)/mse)
    return psnr


def MSE_NSCR_UACI(original, encrypted):
    # show_info("ORIGINAL 1", original)
    # show_info("ENCRYPTED 1", original)
    original = original + 32768
    encrypted = encrypted + 32768
    # show_info("ORIGINAL", original)
    # show_info("ENCRYPTED", original)
    addition = 0
    nscr = 0
    uaci = 0
    # print("MSE PROCEDURE: ")
    for i in range(0, len(original)):
        # print("", i, ":")
        # print("------Sum: ", original[i], "-",
        #       encrypted[i], "=", (original[i] - encrypted[i]))
        # print("------Pow: ", (original[i] - encrypted[i]) ** 2)
        addition = addition + ((int(original[i]) - int(encrypted[i])) ** 2)
        nscr = nscr + (0 if (int(original[i]) == int(encrypted[i])) else 1)
        # if nscr > 0:
        #     print("NSCR", nscr)

        uaci = uaci + \
            ((abs(int(original[i] - encrypted[i])))/(len(original)*65535))
        # print("------Total sum: ", addition)

    # print("N: ", len(original))
    # print("Total sum: ", addition)
    return addition/len(original), (nscr/len(original))*100, (uaci)*100


def SNR_2(original, encrypted):
    original = original + 32768
    encrypted = encrypted + 32768
    denominator = 0
    numerator = 0
    for i in range(0, len(original)):
        numerator = numerator + (int(original[i]) ** 2)
        denominator = denominator + abs(int(original[i]) - int(encrypted[i]))

    # print("Numerator: ", numerator)
    # print("Denominator: ", denominator)

    return 10 * math.log10(numerator/denominator)


def SNR_3(original, encrypted):
    original = original + 32768
    encrypted = encrypted + 32768
    denominator = 0
    numerator = 0
    for i in range(0, len(original)):
        numerator = numerator + (int(original[i]) ** 2)
        denominator = denominator + \
            ((int(original[i]) - int(encrypted[i])) ** 2)

    # print("Numerator: ", numerator)
    # print("Denominator: ", denominator)

    return 10 * math.log10(numerator/denominator)


def get_statics(init_data, final_data):

    # print("Original samples \n", init_data[0:1000])
    # print("Encrypted samples shape \n", final_data.shape)
    # print("Encrypted samples len \n", len(final_data))

    corr, _ = pearsonr(init_data, final_data)
    print('Pearsons correlation: %.9f' % corr)

    corr, _ = spearmanr(init_data, final_data)
    print('Spearmans correlation: %.9f' % corr)

    pd_series = pd.Series(init_data)
    counts = pd_series.value_counts()
    entpy = entropy(counts)
    print('Entropy original file: %.9f' % entpy)
    pd_series = pd.Series(final_data)
    counts = pd_series.value_counts()
    entpy = entropy(counts)
    print('Entropy encrypted file: %.9f' % entpy)

    print('SNR: %.9f' % SNR(final_data))
    # print('SNR_2: %.9f' % SNR_2(init_data, final_data))
    # print('SNR_3: %.9f' % SNR_3(init_data, final_data))
    # print('MSE: %.9f' % np.mean((init_data - final_data) ** 2))
    # print('MSE: %.9f' % (np.sum((init_data - final_data) ** 2) / len(final_data)))
    # print('MSE: %.9f' % (np.subtract(init_data, final_data) ** 2).mean())
    mse, nscr, uaci = MSE_NSCR_UACI(init_data, final_data)
    # print('MSE: %.9f' % mse)
    print('PSNR: %.9f' % PSNR(init_data, final_data, mse))

    comparison = np.array(init_data) == np.array(final_data)
    equal_arrays = comparison.all()
    print("SON IGUALES: ", equal_arrays)
    print('NSCR: %.9f' % nscr)
    print('UACI: %.9f' % uaci)


@profile
def encrypt_data(num_track, custom: bool, c_x, c_y, c_n, c_m, t_u, t_x, l_u, l_x):
    rate, data = scipy.io.wavfile.read(tracks[(num_track - 1)][0])

    # show_info("DATA ORIGIN", data)
    # print("rate:", rate)
    # print("dataset samples:", data[22:54, :])
    # bytes_data = bytes(data[0])

    data_header = data[0:22, :]
    # show_info("HEADER", data_header)
    # print("DATASET HEADER:")
    # print(data[0:22, 0])
    # print("ORIGINAL POSICION 1654:", data[1654, 0])

    # comparison = np.array(data) == np.array(custom_data)
    # equal_arrays = comparison.all()
    # print("SON IGUALES: ", equal_arrays)
    # print("ORIGINAL POSICION 1654:", data[1654, 0])
    # print("CUSTOM POSICION 1654:", custom_data[1654, 0])
    # if custom:
    #     print("1")
    #     # print("ORIGINAL POSICION 1654 ANTES:", data[1654, 0])
    #     custom_data = data[0:]
    #     custom_data[1654, 0] = -1430
    #     # print("ORIGINAL POSICION 1654 DESPUES:", data[1654, 0])

    #     # comparison = np.array(data) == np.array(custom_data)
    #     # equal_arrays = comparison.all()
    #     # print("SON IGUALES: ", equal_arrays)
    #     # print("ORIGINAL POSICION 1654:", data[1654, 0])
    #     # print("CUSTOM POSICION 1654:", custom_data[1654, 0])
    #     data_array, key_array = stack_data(custom_data, 16)
    # else:
    #     print("2")
    #     data_array, key_array = stack_data(data, 16)

    data_array, key_array = stack_data(data, 16, c_x, c_y, c_n, c_m)

    positions_bits, positions_bytes, tail_lenght, block_lenght = permutation_data(
        len(data_array), t_u, t_x, l_u, l_x)

    # print("Cantidad positions: ", len(positions_array))
    # print("Muestra de índices de permutación de vectores de 4-bit:")
    # print(positions_bits[:20])

    # print("Muestra de índices de permutación de bloques:")
    # print(positions_bytes[:20])

    permuted_array = permutate_data(
        data_array, positions_bits, positions_bytes, tail_lenght, block_lenght)
    # print("Muestra permuted data:")
    # print(permuted_array[10:18])

    # print("DATA LEN:", len(permuted_array))
    # print("KEY LEN:", len(key_array))
    xor_array = np.bitwise_xor(permuted_array, key_array)
    # print("- AUDIO ------- CLAVE ------ RESULTADO")
    # print("[1 0 0 0] XOR [1 0 0 0] ---> [1 0 0 0]")
    # for i in range(len(permuted_array)):
    #     if i < 10:
    #         print(permuted_array[i], "XOR", key_array[i], "--->", xor_array[i])
    # show_info("XOR ARRAY", xor_array)
    substituted_array = substitute_data(xor_array)
    # print("Muestra substituted data:")
    # print(data[22:122, 0])
    # print(substituted_array[0:100])

    final_array = convert_to_wav_file_format(
        data, substituted_array, data_header)

    # get_statics(data[22:, 0], final_array[22:, 0])

    # scipy.io.wavfile.write(
    #     'audio/Encrypted/' + tracks[(num_track - 1)][1] + '-encrypted.wav', rate, final_array)

    # show_info("ORIGINAL DATA", data[22:, 0])
    # show_info("ENCRYPTED DATA", final_array[22:, 0])

    # plots(data[22:, 0], "- Original", rate)
    # plots(final_array[22:, 0], "- Encrypted", rate)

    return final_array[22:, 0]


print("Starting encryption....")
main_data = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
                         CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
print("Data encrypted!")
# c_x, c_y, c_n, c_m, t_u, t_x, l_u, l_x = generateSeeds()

# main_data = encrypt_data(2, False, c_x, c_y, c_n, c_m, t_u, t_x, l_u, l_x)
# data_1 = encrypt_data(2, False, c_x + 0.1, c_y, c_n, c_m, t_u, t_x, l_u, l_x)
# data_2 = encrypt_data(2, False, c_x, c_y + 0.1, c_n, c_m, t_u, t_x, l_u, l_x)
# data_3 = encrypt_data(2, False, c_x, c_y, c_n + 0.1, c_m, t_u, t_x, l_u, l_x)
# data_4 = encrypt_data(2, False, c_x, c_y, c_n, c_m + 0.1, t_u, t_x, l_u, l_x)
# data_5 = encrypt_data(2, False, c_x, c_y, c_n, c_m, t_u + 0.2, t_x, l_u, l_x)
# data_6 = encrypt_data(2, False, c_x, c_y, c_n, c_m, t_u, t_x + 0.2, l_u, l_x)
# data_7 = encrypt_data(2, False, c_x, c_y, c_n, c_m, t_u, t_x, l_u + 0.2, l_x)
# data_8 = encrypt_data(2, False, c_x, c_y, c_n, c_m, t_u, t_x, l_u, l_x + 0.2)

# main_data = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                          CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
# data_1 = encrypt_data(1, False, CHEBYSHEV_X_0 + 0.1, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                       CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
# data_2 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0 + 0.1, CHEBYSHEV_N,
#                       CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
# data_3 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N +
#                       0.1, CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
# data_4 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                       CHEBYSHEV_M + 0.1, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
# data_5 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                       CHEBYSHEV_M, TIENDA_U + 0.1, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0)
# data_6 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                       CHEBYSHEV_M, TIENDA_U, TIENDA_X_0 + 0.1, LOGISTICA_U, LOGISTICA_X_0)
# data_7 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                       CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U + 0.1, LOGISTICA_X_0)
# data_8 = encrypt_data(1, False, CHEBYSHEV_X_0, CHEBYSHEV_Y_0, CHEBYSHEV_N,
#                       CHEBYSHEV_M, TIENDA_U, TIENDA_X_0, LOGISTICA_U, LOGISTICA_X_0 + 0.1)


# print("-------------------")
# print("Cambio 1:")
# get_statics(main_data, data_1)
# print("-------------------")
# print("Cambio 2:")
# get_statics(main_data, data_2)
# print("-------------------")
# print("Cambio 3:")
# get_statics(main_data, data_3)
# print("-------------------")
# print("Cambio 4:")
# get_statics(main_data, data_4)
# print("-------------------")
# print("Cambio 5:")
# get_statics(main_data, data_5)
# print("-------------------")
# print("Cambio 6:")
# get_statics(main_data, data_6)
# print("-------------------")
# print("Cambio 7:")
# get_statics(main_data, data_7)
# print("-------------------")
# print("Cambio 8:")
# get_statics(main_data, data_8)
