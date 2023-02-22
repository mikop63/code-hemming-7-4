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

    bit_arr = standart(bit)
    print('Комбинация до добавления ошибки:', )
    for word in bit_arr:
        print(word)
    for i in range(len(bit_arr)):
        bit_arr[i] = hamming_encoding(bit_arr[i])
    print('Кодовая комбинация до добавления ошибки:', )
    for word in bit_arr:
        print(word)

    bit_error = input('В какой разряд внести ошибку (от 1 до 4): ')
    while not re.match("^[1-4]$", bit_error):
        bit_error = input('Я же сказал от 1 до 4!!!\nПопробуй еще раз: ')
    bit_error = int(bit_error)

    bit_arr[0][bit_error - 1] = not bit_arr[0][bit_error - 1]
    print('Кодовая комбинация после добавления ошибки:', bit_arr[0])

    for bit_with_err in bit_arr:
        position = get_error_position(bit_with_err)
        if position is None:
            print("Ошибки нет")
        else:
            print(f'Ошибка в разряде №{position}')
            print(f'Без ошибок: {bit}')
            print(f'С ошибками: {bit_with_err[:4]}')
            bit_with_err[position - 1] = not bit_with_err[position - 1]
            print(f'После исправления: {bit_with_err[:4]}')


if __name__ == '__main__':
    main()
