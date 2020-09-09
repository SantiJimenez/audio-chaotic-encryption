import scipy.io.wavfile
import numpy as np


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

    # Create new array with [[8],[8], ... [8]] shape of binaries_array*2 lenght
    bytes_array = np.zeros((data_array.shape[0]*2), dtype=np.int8)
    bytes_array = np.zeros(list(bytes_array.shape) + [8], dtype=np.int8)

    # Function to return bit to bit to a [16] vector
    vectorize_bits_function = np.vectorize(lambda x: x[i_byte])

    print("|------------------------------------|")
    for i_byte in range(0, m):
        # print("binaries_array.shape: ", binaries_array.shape)
        # print("binaries_array: ", binaries_array)
        # print(i_byte, ": ", vectorize_bits_function(
        #     strings_array).astype("int8"))
        binaries_array[..., i_byte] = vectorize_bits_function(
            strings_array).astype("int16")

    print("binaries_array: ", binaries_array[100:110])

    j = 0
    print("Lenght bytes_array: ", bytes_array.shape[0])

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

    flatten_array = bytes_array.flatten()
    cantidad_binarios_fila = 400
    cantidad_binarios_columna = 400
    cantidad_elementos_totales_matriz = cantidad_binarios_fila * \
        cantidad_binarios_columna  # 400 * 400
    num_matrices = len(flatten_array) // cantidad_elementos_totales_matriz

    matrices_datos_crudos = []
    el = 0
    print("Numero de matrices {}".format(num_matrices))
    for i in range(0, num_matrices):
        matriz = []
        for r in range(0, cantidad_binarios_fila):
            row = []
            for c in range(0, cantidad_binarios_columna):
                row.append(flatten_array[el])
                el += 1
            matriz.append(row)
        matrices_datos_crudos.append(matriz)
        print("size matrices w ={}x{}".format(len(matriz[0]), len(matriz)))

    cola = []
    for j in range(el, len(flatten_array)):
        cola.append(flatten_array[j])
    print("Len cola = {}".format(len(cola)))

    print("matrices muestras: ", matrices_datos_crudos[0][10:20])

    return bytes_array


data_matrix = stack_data(data, 16)

# print("data_matrix: ", data_matrix)
