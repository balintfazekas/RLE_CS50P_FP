import pytest
import numpy as np
from csv import DictReader
from PIL import Image
from project import (
    get_image_paths,
    read_image_as_array,
    is_binary,
    create_label_dictionary,
    label_array,
    make_substring,
    rle_encoding,
    rle_decoding,
    is_csv,
    Encoder,
    Decoder
)

# CONSTANTS
BINARY_ARRAY = np.array([ [0,0,0], [1,1,1] ])
NON_BINARY_ARRAY = np.array([ [0,0,0], [2,1,1] ])


def test_is_binary():
    assert is_binary(BINARY_ARRAY) == True
    assert is_binary(NON_BINARY_ARRAY) == False


def test_create_label_dictionary():
    assert create_label_dictionary(BINARY_ARRAY, 'standard') == { 0 : 'b', 1 : 'w'}
    assert create_label_dictionary(BINARY_ARRAY, 'inverse') == { 0 : 'w', 1 : 'b'}

    with pytest.raises(ValueError):
        assert create_label_dictionary(BINARY_ARRAY, 'nonvalid')


def test_label_array():
    input_array = np.reshape(BINARY_ARRAY, -1)
    standard = create_label_dictionary(input_array, 'standard')
    inverse = create_label_dictionary(input_array, 'inverse')

    assert np.array_equal(label_array(input_array, standard),
                          np.array(['b','b','b', 'w','w','w']))
    assert np.array_equal(label_array(input_array, inverse),
                          np.array(['w','w','w','b','b','b']))
    

def test_make_substring():
    assert make_substring(1, 'a', 2) == 'a'
    assert make_substring(2, 'a', 2) == 'aa'
    assert make_substring(3, 'a', 2) == '3a'
    assert make_substring(3, 'a', 3) == 'aaa'


def test_rle_encoding():
    input_array = [char for char in 'abbcccdddd']
    assert rle_encoding(input_array) == 'abb3c4d'


def test_rle_decoding():
    assert np.array_equal(rle_decoding('abb3c4d'),
                          [char for char in 'abbcccdddd'])
    

def test_is_csv():
    assert is_csv('some/example/path/wit/csvfile.csv') == True

    with pytest.raises(ValueError):
        assert is_csv('some/example/path/without/csvfile.tiff')
