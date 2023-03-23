from bitarray import bitarray
import random
import time
import numpy as np
from numba import njit


@njit
def generate_random_bit(length, probability_of_one):
    """
    Функция, которая генерирует случайную последоватльность битов заданной длины
    с заранее заданной вероятностью появляется единица.

    :param length: Длина битового вектора.
    :param probability_of_one: Вероятность появления единицы (1) в битовом векторе (от 0 до 1).
    :return: Сгенерированная последовательность.
    """
    random_numbers = np.random.rand(length)
    bits = ['1' if x < probability_of_one else '0' for x in random_numbers]
    return ''.join(bits)


def hamming_encod(message):
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
    Симуляция канала передачи сообщения.
    При передачи сообщения в канале возникают ошибки

    :param bit: Входное сообщение из длинной последовательности битов.
    :param err_probability: Вероятность ошибки.
    Вероятность должна быть маленькая, чтобы в каждой последовательности было не более 1 ошибки
    :return bit: Последовательность входных битов вместе с проверочными.
    :return real_error_count: Колличество допущенных ошибок, чтобы в конце провеить сколько нашли и исправили.
    """
    real_error_count = 0
    for i in range(len(bit)):
        if random.random() < err_probability:  # с вероятностью err_probability генерируем 1
            bit[i] = not bit[i]
            real_error_count += 1

    return bit, real_error_count


def hamming_decod(received_code):
    """
    Получение места ошибки в последовательности принятых битов
    Исправление этой ошибки

    :param received_code: Входное сообщение из последовательности битов.
    :return error_position: Исправленное сообщение.
    """
    syndrome_table = {
        0b101: 0,
        0b110: 1,
        0b111: 2,
        0b011: 3,
        0b100: 4,
        0b010: 5,
        0b001: 6
    }

    syndrome = ((received_code[4] ^ received_code[0] ^ received_code[1] ^ received_code[2]) << 2) | \
               ((received_code[5] ^ received_code[1] ^ received_code[2] ^ received_code[3]) << 1) | \
               (received_code[6] ^ received_code[0] ^ received_code[2] ^ received_code[3])
    error_position = syndrome_table.get(syndrome)
    if error_position is not None:
        received_code[error_position] = not received_code[error_position]
    return received_code


def hamming_get_data(brx):
    """
    Получение информационных битов из Кода Хемминга

    :param brx: Получение 7 битов. 4 информационных + 3 проверочных
    :return: 4 информационных бита
    """
    a = brx
    first_four = a[:4]
    return first_four

def compareVectors(a, b):
    """
    Подсчет количества ошибок

    :return: число отличающихся бит между a и b
    """
    count = 0
    # переводим к одному типу
    if isinstance(a, str):
        a = bitarray(a)
    if isinstance(b, str):
        b = bitarray(b)

    for i in range(len(a)):
        c = a[i] ^ b[i]
        count += c

    return count


def main():
    # length = int(input('Введите колличество блоков: '))
    length = 1000000
    probability_of_one = 0.8

    # length = 10000
    err_probabilitys = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]
    NErrBlocks = [0] * len(err_probabilitys)
    NErrBits = [0] * len(err_probabilitys)
    real_error_count = [0] * len(err_probabilitys)
    # tx - переданные (transmitted)
    # rx - принятые (received)
    for i in range(length):
        atx = generate_random_bit(4, probability_of_one)
        btx = hamming_encod(atx)
        for i in range(len(err_probabilitys)):
            c, errors = channel_simulation(btx.copy(), err_probabilitys[i]) # если не передаваить btx.copy, тогда переменная btx перезапишется
            real_error_count[i] += errors # получаем реальное количество ошибок
            brx = hamming_decod(c)
            arx = hamming_get_data(brx)
            # Вероятность ошибки (P) по оси Х
            # частость ошибки в канале
            # частость ошибки после декадирования (P*) по оси Y
            NErrBlocks[i] += compareVectors(btx, brx)
            NErrBits[i] += compareVectors(atx, arx)

    print(f'Колчисетво битов {length*4}')
    print(f'Теоретическая вероятность ошибки: {err_probabilitys}')
    print(f'Частость ошибки в канале: {real_error_count}')
    print(f'Частость ошибки после декодирования: {NErrBits}')

    for i in range(len(NErrBits)):
        NErrBits[i] /= length*4
    print(f'Реальная вероятность ошибки в канале: {real_error_count}')
    print(f'Вероятность ошибки после декодирования: {NErrBits}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration_in_seconds = end_time - start_time
    minutes, seconds = divmod(duration_in_seconds, 60)
    print(f"Программа выполнялась за: {int(minutes)} мин. {int(seconds)} сек.")