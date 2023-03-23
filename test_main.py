import unittest
from main import *


class TestHammingCode(unittest.TestCase):

    def test_hamming_encoding(self):
        bit_array = [
            bitarray('1010'),
            bitarray('1111'),
            bitarray('0000'),
            bitarray('0101')
        ]
        expected_encoded_array = [
            bitarray('1010010'),
            bitarray('1111111'),
            bitarray('0000000'),
            bitarray('0101101')
        ]

        # Проверяем правильность кодирования каждой строки
        for i in range(len(bit_array)):
            self.assertEqual(expected_encoded_array[i], hamming_encod(bit_array[i]))

    def test_get_error_position(self):
        bit_with_err = [
            bitarray('1111001'),
            bitarray('1111111'),
            bitarray('1110000'),
            bitarray('1001000'),
            bitarray('1111110')
        ]
        expected_position = [
            2, None, 5, 2, 7
        ]

        for i in range(len(bit_with_err)):
            self.assertEqual(expected_position[i], hamming_decod(bit_with_err[i]))

    # def test_standart(self):
    #     bit = '101101111101110111111111011101101111101111011011111101011110011101101111111111111011111110011111010111111111101101111011001001111111111111001111111111'
    #     expected_bit_array = ['1011', '0111', '1101', '1101', '1111', '1111',
    #                           '0111', '0110', '1111', '1011', '1101', '1011',
    #                           '1111', '0101', '1110', '0111', '0110', '1111',
    #                           '1111', '1111', '1011', '1111', '1001', '1111',
    #                           '0101', '1111', '1111', '1011', '0111', '1011',
    #                           '0010', '0111', '1111', '1111', '1100', '1111', '1111', '0011']
    #     self.assertEqual(expected_bit_array, standart(bit))