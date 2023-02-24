from bitarray import bitarray
import random
import time
import numpy as np
from numba import njit


@njit
def generate_random_bit(length, probability_of_one):
    """
    Функция, которая генерирует случайный битовый вектор заданной длины с заданной вероятностью
    появления единицы (1) в нем.

    :param length: Длина битового вектора.
    :param probability_of_one: Вероятность появления единицы (1) в битовом векторе (от 0 до 1).
    :return: Сгенерированная последовательность.
    """
    random_numbers = np.random.rand(length)
    bits = ['1' if x < probability_of_one else '0' for x in random_numbers]
    return ''.join(bits)


@njit
def standart(bit):
    """
    разбиваем длинную строку на биты по 4 штуки
    """
    return [bit[i:i + 4].zfill(4) for i in range(0, len(bit), 4)]


@njit(cache=True)
def generate_bit_from_str(text):
    """
    Функция, которая преобразует биты в строку.

    :param text: Входной текст который преобразуется в строку.
    :return: Сгенерированная последовательность.
    """
    binary_string = ''.join([format(ord(c), '08b') for c in text])
    return binary_string


def hamming_encoding(message):
    """
    Функция, которая выполняет кодирование Хемминга (7,4)

    :param message: Входное сообщение из битов.
    :return: Последовательность входных битов вместе с проверочными.
    """
    # Исходный входной битовый вектор
    input_bits = bitarray(message)
    output_bits = input_bits[:4]  # копирование первых 4 битов из входного вектора
    output_bits.append(input_bits[0] ^ input_bits[1] ^ input_bits[2])  # вычисление 5-го бита
    output_bits.append(input_bits[1] ^ input_bits[2] ^ input_bits[3])  # вычисление 6-го бита
    output_bits.append(input_bits[0] ^ input_bits[2] ^ input_bits[3])  # вычисление 7-го бита
    return output_bits


def channel_simulation(bit, err_probability):
    """
    Симуляция канала передачи сообщения

    :param bit: Входное сообщение из длинной последовательности битов.
    :param err_probability: Вероятность ошибки.
    Вероятность должна быть маленькая, чтобы в каждой последовательности было не более 1 ошибки
    :return bit: Последовательность входных битов вместе с проверочными.
    :return error_count: Колличество допущенных ошибок, чтобы в конце провеить сколько нашли и исправили.
    """
    error_count = 0
    for i, bit_elem in enumerate(bit):
        if random.random() < err_probability:  # с вероятностью probability_of_one генерируем 1
            bit[i] = not bit[i]
            error_count += 1
    return bit, error_count


def get_error_position(received_code):
    """
    Получение места ошибки в последовательности

    :param received_code: Входное сообщение из последовательности битов.
    :return error_position: Место ошибки в этой последовательности.
    """
    syndrome_table = {
        0b101: 1,
        0b110: 2,
        0b111: 3,
        0b011: 4,
        0b100: 5,
        0b010: 6,
        0b001: 7
    }

    syndrome = ((received_code[4] ^ received_code[0] ^ received_code[1] ^ received_code[2]) << 2) | \
               ((received_code[5] ^ received_code[1] ^ received_code[2] ^ received_code[3]) << 1) | \
               (received_code[6] ^ received_code[0] ^ received_code[2] ^ received_code[3])
    error_position = syndrome_table.get(syndrome)
    return error_position if error_position is not None else None  # если ошибки нет, вернется None


def main():
    # length = int(input('Введите длину последовательности: '))
    length = 40000000
    # probability_of_one = float(input('Введите вероятность появления 1: '))
    probability_of_one = 0.8
    # TODO: добавить возсожно ввода текста. Функция уже написанна

    bit = generate_random_bit(length, probability_of_one)

    # засекаем время
    end_time = time.time()
    duration_in_seconds = end_time - start_time
    minutes, seconds = divmod(duration_in_seconds, 60)
    print(f"Генерация: {int(minutes)} мин. {int(seconds)} сек.")

    # Комбинация до добавления ошибки
    bit_array = standart(bit)
    bit_arr_encode = bit_array.copy()  # копия bit_array, в которую отправим в канал
    # print('Комбинация до добавления ошибки:', )
    # [print(word) for word in bit_array]
    bit_arr_encode = [hamming_encoding(bit) for bit in bit_array]
    # print('Кодовая комбинация до добавления ошибки:', )
    # [print(word) for word in bit_arr_encode]

    end_time = time.time()
    duration_in_seconds = end_time - start_time
    minutes, seconds = divmod(duration_in_seconds, 60)
    print(f"Кодирование Хемминга выполнялась: {int(minutes)} мин. {int(seconds)} сек.")

    # для канала передачи все преобразуем в одну длинную последовательность
    bit_code = bitarray(len(bit_arr_encode) * 7)
    for i, bit in enumerate(bit_arr_encode):
        start = i * 7
        end = (i + 1) * 7
        bit_code[start:end] = bit

    err_probability = 0.01
    bit_long_str_with_error, error_total_count = channel_simulation(bit_code,
                                                                    err_probability)  # кодовая комбинация после добавления ошибки

    end_time = time.time()
    duration_in_seconds = end_time - start_time
    minutes, seconds = divmod(duration_in_seconds, 60)
    print(f"Симуляция канала: {int(minutes)} мин. {int(seconds)} сек.")

    # print('Кодовая комбинация после добавления ошибки:', bit_long_str_with_error)
    # получаем массив с элементоми по 7 байт в которых есть ошибка
    bit_err_arr = [bit_long_str_with_error[i:i + 7] for i in range(0, len(bit_long_str_with_error), 7)]

    position = 0
    err_found_count = 0
    for bit_with_err in bit_err_arr:
        position_err = get_error_position(bit_with_err)
        if position_err is None:
            # print(f"{position}). Ошибки нет")
            pass
        else:
            err_found_count += 1
            if length < 15:
                print(
                    f'\n{position}). Ошибка в разряде №{position_err} {"В проверочном бите" if position_err > 4 else ""}')
                print(f'Без ошибоки:{bit_array[position]}')
                print(f'С ошибкой: \t{bit_with_err[:4].to01()}')
                bit_with_err[int(position_err) - 1] = not bit_with_err[int(position_err) - 1]
                print(f'После исправления: {bit_with_err[:4].to01()}')
        position += 1
    print(f'\nПередано: {length} бит')
    print(f'Передано комбинаций: {position - 1}')
    print('Допущено ошибок:', error_total_count)
    print('Найдено ошибок:', err_found_count)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration_in_seconds = end_time - start_time
    minutes, seconds = divmod(duration_in_seconds, 60)
    print(f"Программа выполнялась за: {int(minutes)} мин. {int(seconds)} сек.")
