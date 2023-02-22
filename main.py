from BitVector import *
import re
import random


def generate_random_bit(length, probability_of_one):
    """
    Функция, которая генерирует случайный битовый вектор заданной длины с заданной вероятностью
    появления единицы (1) в нем.

    :param length: Длина битового вектора.
    :param probability_of_one: Вероятность появления единицы (1) в битовом векторе (от 0 до 1).
    :return: Сгенерированная последовательность.
    """
    random_string = ''
    for i in range(length):
        if random.random() < probability_of_one:  # с вероятностью probability_of_one генерируем 1
            random_string += '1'
        else:  # с вероятностью 1-probability_of_one генерируем 0
            random_string += '0'
    return random_string


def generate_bit_from_str(text):
    """
    Функция, которая преобразует биты в строку.

    :param text: Входной текст который преобразуется в строку.
    :return: Сгенерированная последовательность.
    """
    binary_string = ''.join([format(ord(c), '08b') for c in text])
    return binary_string


def hamming_encoding(message):
    '''
    Функция, которая выполняет кодирование Хемминга (7,4)

    :param message: Входное сообщение из битов.
    :return: Последовательность входных битов вместе с проверочными.
    '''
    # Исходный входной битовый вектор
    input_bits = BitVector(bitstring=message)
    output_bits = BitVector(size=7)
    # Добавление проверочных битов
    output_bits[0] = input_bits[0]
    output_bits[1] = input_bits[1]
    output_bits[2] = input_bits[2]
    output_bits[3] = input_bits[3]
    output_bits[4] = input_bits[0] ^ input_bits[1] ^ input_bits[2]
    output_bits[5] = input_bits[1] ^ input_bits[2] ^ input_bits[3]
    output_bits[6] = input_bits[0] ^ input_bits[2] ^ input_bits[3]

    return output_bits


def channel_simulation(bit, err_probability):
    for i in range(bit.length()):
        if random.random() < err_probability:  # с вероятностью probability_of_one генерируем 1
            bit[i] = not bit[i]
    return bit


def get_error_position(received_code):
    syndrome_table = {
        "101": 1,
        "110": 2,
        "111": 3,
        "011": 4,
        "100": 5,
        "010": 6,
        "001": 7
    }

    syndrome = BitVector(size=3)
    syndrome[0] = received_code[4] ^ received_code[0] ^ received_code[1] ^ received_code[2]
    syndrome[1] = received_code[5] ^ received_code[1] ^ received_code[2] ^ received_code[3]
    syndrome[2] = received_code[6] ^ received_code[0] ^ received_code[2] ^ received_code[3]
    error_position = syndrome_table.get(str(syndrome))

    if error_position is None:
        error_position = "Error: No error detected."
        return None
    return error_position


def standart(bit):
    """
    разбиваем длинную строку на биты по 4 штуки
    """
    bit_arr = []
    for i in range(0, len(bit), 4):  # диапазон от 0 до длины массива с шагом 4
        bit_word = bit[i:i + 4]      # собираем в слова по 4 символа
        """
        Дописываем вначало недостоющие биты. Чтобы стало 4.
        Например у нас было "101", а станет "0101" 
        """
        while len(bit_word) < 4:
            bit_word = '0' + bit_word
        bit_arr.append(bit_word)
    return bit_arr


def main():
    length = int(input('Введите длину последовательности: '))
    probability_of_one = float(input('Введите вероятность появления 1: '))
    # TODO: добавить возсожно ввода текста. Функция уже написанна
    # probability_of_one = 0.8
    while not probability_of_one < 1:
        probability_of_one = input('Введите вероятность появления 1 (например 0.6)')
    bit = generate_random_bit(length, probability_of_one)

    bit_array = standart(bit) # Комбинация до добавления ошибки
    bit_arr_encode = bit_array.copy()
    # print('Комбинация до добавления ошибки:', )
    # [print(word) for word in bit_array]
    for i in range(len(bit_arr_encode)):
        bit_arr_encode[i] = hamming_encoding(bit_arr_encode[i]) # Кодовая комбинация до добавления ошибки
    # print('Кодовая комбинация до добавления ошибки:', )
    # [print(word) for word in bit_arr_encode]


    # для канала передачи все преобразуем в одну длинную последовательность
    bit_code = BitVector(size=len(bit_arr_encode) * 7)
    for i, bit_vector in enumerate(bit_arr_encode):
        bit_code[i * 7:(i + 1) * 7] = bit_vector

    err_probability = 0.01
    bit_long_str_with_error = channel_simulation(bit_code, err_probability) # кодовая комбинация после добавления ошибки

    # print('Кодовая комбинация после добавления ошибки:', bit_long_str_with_error)
    bit_err_arr = []

    for i in range(0, len(bit_long_str_with_error), 7):
        bit_err_word = bit_long_str_with_error[i:i + 7]
        bit_err_arr.append(bit_err_word)                    # получаем массив с элементоми по 7 байт в которых есть ошибка
    # print(bit_arr_encode[0])
    # print(bit_err_arr[0])
    position = 0
    for bit_with_err in bit_err_arr:
        position_err = get_error_position(bit_with_err)
        if position_err is None:
            print(f"{position}). Ошибки нет")
        else:
            print(f'{position}). Ошибка в разряде №{position_err} {"В проверочном бите" if position_err > 4 else ""}')
            print(f'Без ошибоки:{bit_array[position]}')
            print(f'С ошибкой: \t{bit_with_err[:4]}')
            bit_with_err[position_err - 1] = not bit_with_err[position_err - 1]
            print(f'После исправления: {bit_with_err[:4]}')
        position += 1


if __name__ == '__main__':
    main()
