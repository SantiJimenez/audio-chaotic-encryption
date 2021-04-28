#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import numpy as np
import math
# from FrequencyTest import FrequencyTest
# from RunTest import RunTest
# from Matrix import Matrix
# from Spectral import SpectralTest
# from TemplateMatching import TemplateMatching
# from Universal import Universal
# from Complexity import ComplexityTest
# from Serial import Serial
# from ApproximateEntropy import ApproximateEntropy
# from CumulativeSum import CumulativeSums
# from RandomExcursions import RandomExcursions

# Import src

from nistrng import *


# --------- VARIABLES DE CONTROL ---------- #
#
# Polinomios de Chebyshev
# x0, y0, m ,n
CHEBYSHEV_X_0 = 0.4811295983708204
CHEBYSHEV_Y_0 = 0.3437604907376320
CHEBYSHEV_N = 5.0468764305551317
CHEBYSHEV_M = 9.0078127384258552

# Funcion Tienda
# a
TIENDA_U = 1.5546548465465465
TIENDA_X_0 = 0.5811295983708204

# Funcion Logistica
# u
LOGISTICA_U = 3.7115645816513156
LOGISTICA_X_0 = 0.6734678425981357
# ---------- VARIABLES DE CONTROL ---------- #


def normalizationValues(x):
    n1 = math.floor(x * 10**5)
    n2 = math.floor(x * 10**9) - (n1 * 10**4)
    return (n1 ^ n2) % 256


def polynomialSequence(x, m):
    return math.cos(m*math.acos(x))


def generateSeeds():
    # np.random.seed(42)
    np.set_printoptions(precision=16)
    c_m = np.random.uniform(2, 20)
    c_n = np.random.uniform(2, 20)
    c_x = np.random.uniform(-1, 1)
    c_y = np.random.uniform(-1, 1)

    # t_u = np.random.uniform(0, 1)
    t_u = np.random.uniform(1, 2)
    t_x = np.random.uniform(0, 1)

    l_u = 3.5778654516456545
    l_x = np.random.uniform(0, 1)
    # l_u = 3.7115645816513156
    # l_x = 0.6734678425981357

    print("     CM:", c_m)
    print("     CN:", c_n)
    print("     CX:", c_x)
    print("     CY:", c_y)
    print("     TU", t_u)
    print("     TX", t_x)
    print("     LU", l_u)
    print("     LX", l_x)

    return c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x


def int_to_binary_array(num, m):
    # Convert a positive integer num into an m-bit bit vector
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def generateData(lenght, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x):

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
    list_keys = []

    m = c_m
    n = c_n
    s_i = c_x
    t_i = c_y
    x_i = t_x
    y_i = l_x

    for i in range(0, lenght + 51):
        x_i = tend_map_iterator(x_i)
        # y_i = logistic_map_iterator(y_i)

        if i > 50:
            ## x_final = math.floor((x_i * 10**2) % 4)
            # list_positions_tent.append(x_final)
            # print("x_i: ", x_i)
            list_positions_tent.append(
                int_to_binary_array(normalizationValues(x_i), 8))

            ## y_final = math.floor((y_i * 10**2) % 4)
            # list_positions_logistic.append(y_final)
            # print("y_i: ", y_i)
            # list_positions_logistic.append(normalizationValues(y_i))

        # if i < int(lenght/2):
        #     s_i = polynomialSequence(s_i, m)
        #     t_i = polynomialSequence(t_i, n)
        #     list_keys.append(normalizationValues(s_i))
        #     list_keys.append(normalizationValues(t_i))

    # return list_keys, list_positions_tent, list_positions_logistic
    return list_positions_tent


def test(sequence_name, sequence_data, results_count):
    print("|--------------------------------|", sequence_name,
          "|--------------------------------------------|")
    print("Random", sequence_name, "sequence lenght:")
    print(len(sequence_data))
    # Check the eligibility of the test and generate an eligible battery from the default NIST-sp800-22r1a battery
    eligible_battery: dict = check_eligibility_all_battery(
        sequence_data, SP800_22R1A_BATTERY)
    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        # print("-" + name)
        # Test the sequence on the eligible tests
        results = run_all_battery(
            sequence_data, eligible_battery, False)
        # Print results one by one

    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(np.round(result.score, 16)) + " - " +
                  result.name + " - elapsed time: " + str(elapsed_time) + " ms")
            results_count[result.name] = results_count[result.name] + 1
        else:
            print("- FAILED - score: " + str(np.round(result.score, 3)) + " - " +
                  result.name + " - elapsed time: " + str(elapsed_time) + " ms")

    return results_count


def main():

    results_count_c = {
        "Monobit": 0,
        "Frequency Within Block": 0,
        "Runs": 0,
        "Longest Run Ones In A Block": 0,
        "Binary Matrix Rank": 0,
        "Discrete Fourier Transform": 0,
        "Non Overlapping Template Matching": 0,
        "Overlapping Template Matching": 0,
        "Maurers Universal": 0,
        "Linear Complexity": 0,
        "Serial": 0,
        "Approximate Entropy": 0,
        "Cumulative Sums": 0,
        "Random Excursion": 0,
        "Random Excursion Variant": 0
    }

    results_count_t = {
        "Monobit": 0,
        "Frequency Within Block": 0,
        "Runs": 0,
        "Longest Run Ones In A Block": 0,
        "Binary Matrix Rank": 0,
        "Discrete Fourier Transform": 0,
        "Non Overlapping Template Matching": 0,
        "Overlapping Template Matching": 0,
        "Maurers Universal": 0,
        "Linear Complexity": 0,
        "Serial": 0,
        "Approximate Entropy": 0,
        "Cumulative Sums": 0,
        "Random Excursion": 0,
        "Random Excursion Variant": 0
    }

    results_count_l = {
        "Monobit": 0,
        "Frequency Within Block": 0,
        "Runs": 0,
        "Longest Run Ones In A Block": 0,
        "Binary Matrix Rank": 0,
        "Discrete Fourier Transform": 0,
        "Non Overlapping Template Matching": 0,
        "Overlapping Template Matching": 0,
        "Maurers Universal": 0,
        "Linear Complexity": 0,
        "Serial": 0,
        "Approximate Entropy": 0,
        "Cumulative Sums": 0,
        "Random Excursion": 0,
        "Random Excursion Variant": 0
    }

    for i in range(0, 1):
        print("||---------------- ITERATION", i, "----------------------||")
        c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x = generateSeeds()
        # chebyshev_sequence, tent_sequence, logistic_sequence = generateData(
        #     12500, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x)
        # chebyshev_sequence = generateData(
        #     1000, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x)
        # binary_chebyshev_sequence: np.ndarray = pack_sequence(
        #     chebyshev_sequence)

        tent_sequence = generateData(
            12, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x)
        # logistic_sequence = generateData(
        #     12500, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x)
        # binary_tent_sequence: np.ndarray = pack_sequence(
        #     tent_sequence)
        # logistic_sequence = generateData(
        #     10, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x)
        # print(logistic_sequence)
        # binary_logistic_sequence: np.ndarray = pack_sequence(
        #     logistic_sequence)
        # print(binary_logistic_sequence)
        # binary_tent_sequence: np.ndarray = pack_sequence(
        #     tent_sequence)
        # binary_logistic_sequence: np.ndarray = pack_sequence(
        #     logistic_sequence)

        # print("CHEBYSHEV: ", chebyshev_sequence)
        # print(binary_chebyshev_sequence)
        flat_list = [item for sublist in tent_sequence for item in sublist]
        print("TENT: ", flat_list)
        # print(binary_tent_sequence)
        # print("LOGISTIC: ", logistic_sequence)
        # print(binary_logistic_sequence)

        # results_count_c = test(
        #     "CHEBYSHEV", binary_chebyshev_sequence, results_count_c)
        # results_count_t = test(
        #     "TENT", binary_tent_sequence, results_count_t)
        # results_count_l = test(
        #     "LOGISTIC", binary_logistic_sequence, results_count_l)
        # results_count = test(
        #     "TENT", binary_tent_sequence, results_count)
        # results_count = test(
        #     "LOGISTIC", binary_logistic_sequence, results_count)
        # test("TENT", binary_tent_sequence)
        # test("LOGISTIC", binary_logistic_sequence)
        # print("RESULTADOS C: \n", results_count_c)
        # print("RESULTADOS T: \n", results_count_t)
        # print("RESULTADOS L: \n", results_count_l)

    # for name in results_count.keys():
    #   print("- PASSED - score: " + str(np.round(result.score, 16)) + " - " +
    #         result.name + " - elapsed time: " + str(elapsed_time) + " ms")

    # c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x = generateSeeds()

    # Generate the sequence of integers and pack it in its 8-bit representation
    # sequence: np.ndarray = np.random.randint(-128, 128, 1000, dtype=int)
    # binary_sequence: np.ndarray = pack_sequence(sequence)

    # chebyshev_sequence, tent_sequence, logistic_sequence = generateData(
    #     100, c_m, c_n, c_x, c_y, t_u, t_x, l_u, l_x)
    # print("Random Logistic sequence generated by NumPy:")
    # print(logistic_sequence)
    # print("Random Tent sequence generated by NumPy:")
    # print(tent_sequence)

    # binary_chebyshev_sequence: np.ndarray = pack_sequence(chebyshev_sequence)
    # print("Random Chebyshev sequence generated by NumPy encoded in 8-bit signed format:")
    # print(binary_chebyshev_sequence)

    # binary_logistic_sequence: np.ndarray = pack_sequence(logistic_sequence)
    # print("Random Logistic sequence generated by NumPy encoded in 8-bit signed format:")
    # print(binary_logistic_sequence)

    # binary_tent_sequence: np.ndarray = pack_sequence(tent_sequence)
    # print("Random Tent sequence generated by NumPy encoded in 8-bit signed format:")
    # print(binary_tent_sequence)

    # Print sequence
    # print("Random sequence generated by NumPy:")
    # print(sequence)
    # print("Random sequence generated by NumPy encoded in 8-bit signed format:")
    # print(binary_sequence)
    # print("Original sequence taken back by unpacking (to check the correctness of packing process:")
    # print(unpack_sequence(binary_sequence))
    # print("Random Chebyshev sequence generated by NumPy:")
    # print(chebyshev_sequence)
    # print("|----------------------------------------------------------------------------|")
    # print("Random Chebyshev sequence generated by NumPy encoded in 8-bit signed format:")
    # print(binary_chebyshev_sequence)

    # print("Original Chebyshev sequence taken back by unpacking (to check the correctness of packing process:")
    # print(unpack_sequence(binary_chebyshev_sequence))
    # test("CHEBYSHEV", binary_chebyshev_sequence)
    # test("TENT", binary_tent_sequence)
    # test("LOGISTIC", binary_logistic_sequence)


main()
