import glob
import os
import numpy as np
from PIL import Image
from collections import deque
import csv
import sys
import json
from csv import DictWriter, DictReader


def get_image_paths(input_folder: str, suffix: str = '.tif') -> list[str]:
    """Retrieve image file paths from the specified folder.

    Args:
        input_folder (str): The path to the folder containing the images.
        suffix (str): The file extension to filter images (default: '.tif').

    Returns:
        list[str]: A list of file paths that match the specified suffix.
    """
    return glob.glob(os.path.join(input_folder, f"*{suffix}"))


def read_image_as_array(path: str) -> np.ndarray:
    """Read an image file and convert it to a grayscale. Return as NumPy array.

    Args:
        path (str): The file path of the image to be read.

    Returns:
        np.ndarray: A NumPy array representation of the grayscale image.
    """
    return np.array(Image.open(path).convert('L'))


def is_binary(image: np.ndarray) -> bool:
    """Check if a given image is binary.

    Args:
        image (np.ndarray): The image represented as a NumPy array.

    Returns:
        bool: True if the image contains exactly two unique values, False otherwise.
    """
    return len(np.unique(image)) == 2


def image_to_1d_array(image: np.ndarray) -> np.ndarray:
    return np.reshape(image, -1)


def create_label_dictionary(array: np.ndarray, method: str = 'standard') -> str:

    match method:
        case 'standard':
            labels = {
                np.unique(array)[0] : 'b',
                np.unique(array)[1] : 'w',
            }
            return labels

        case 'inverse':
            labels = {
                np.unique(array)[0] : 'w',
                np.unique(array)[1] : 'b',
            }
            return labels

        case _:
            raise ValueError("Method must be 'standard' or 'inverse'.")


def label_array(array : np.ndarray, labels: dict) -> np.ndarray:
    return np.vectorize(labels.get)(array)


def make_substring(counter: int, char: str, threshold: int = 2) -> str:
    if counter <= threshold:
        substring = counter * char
    else:
        substring = f"{counter}{char}"
    
    return substring


def rle_encoding(array: np.ndarray) -> str:
    
    array = deque(array)
    output = ""
    counter = 1
    char = array.popleft()

    while len(array) > 0:
        current_char = array.popleft()

        if len(array) > 0: 
            if current_char != char and len(array) > 0:
                output += make_substring(counter, char)
                char = current_char
                counter = 1          
            else:
                counter += 1

        else:
            if current_char != char:
                output += make_substring(counter, char) + current_char
            else:
                output += make_substring(counter + 1, char)

    return output


def rle_decoding(input: str) -> np.ndarray:
    output = ""
    multiplier = ""
    for char in input:
        if char.isnumeric():
            multiplier += char
        elif not char.isnumeric() and multiplier != "":
            output += int(multiplier) * char
            multiplier = ""
        else:
            output += char
    
    return np.array([char for char in output])


def is_csv(path: str) -> bool:
    
    _, extension = os.path.splitext(path)

    if extension == '.csv':
        return True
    else:
        raise ValueError("Input file is not '.csv' file.")


class Encoder():
    def __init__(self,
                 input_folder: str,
                 output_csv: str = './output.csv',
                 image_suffix: str = '.tif' 
                 ):

        self.input_folder = input_folder
        self.output_csv = output_csv
        self.image_suffix = image_suffix

        # Start encoding automatically 
        self.forward()

    def forward(self, log_excluded = True, labels_method = 'standard'):
        print("Collect image paths...")
        paths = get_image_paths(self.input_folder, self.image_suffix)
        excluded_files = []

        print(f"Create {self.output_csv} file ...")
        with open(self.output_csv, 'w') as file:
            fieldnames = ['Image', 'Shape', 'Labels', 'RLE']
            writer = DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            print("Start RLE encoding ...")
            for index, path in enumerate(paths):

                print(f"Encodeing {index + 1}th image / {len(paths)} images.", end = '\r')
                name = os.path.basename(path)
                img = read_image_as_array(path)

                if not is_binary(img):
                    excluded_files.append(name)
                    continue
                else:
                    shape = img.shape
                    img = np.reshape(img, -1)
                    labels = create_label_dictionary(img, labels_method)
                    img = label_array(img, labels)
                    rle = rle_encoding(img)
                    writer.writerow({
                        'Image' : name,
                        'Shape' : shape,
                        'Labels' : labels,
                        'RLE' : rle
                    })

        if log_excluded and len(excluded_files) > 0:
            with open('excluded.txt', 'w') as file:
                file.writelines(excluded_files)
        
        print("\nCompression is done.\n")


class Decoder():
    def __init__(self,
                 input_file: str,
                 output_folder: str = './reconstructed_masks',
                 image_suffix: str = '.tif'
                 ):
        
        self.input_file = input_file
        self.output_folder = output_folder
        self.image_suffix = image_suffix

        self.forward()


    def forward(self,):

        print("Check input file ...")
        is_csv(self.input_file)
        print("Create output directory ...")
        os.makedirs(self.output_folder, exist_ok = True)

        # need to set it to load 'RLE' field
        csv.field_size_limit(sys.maxsize)
        print(f"Load {self.input_file} ...")
        with open(self.input_file, 'r') as file:
            reader = DictReader(file)
            rows = sum([1 for row in reader])

        # reader need to reinitalise after count rows for some reason.
        with open(self.input_file, 'r') as file:       
            reader = DictReader(file)

            print("Start RLE decoding ...")
            for index, row in enumerate(reader):
                print(f"Encodeing {index + 1}th image / {rows} images.", end = '\r')
                name, _ = os.path.splitext(row['Image'])
                output_path = os.path.join(self.output_folder,
                                           name + self.image_suffix)

                labels = eval(row['Labels'])
                shape = eval(row['Shape'])
                rle = row['RLE']

                decoded_img = rle_decoding(rle)
                switched_labels = {value : key for key, value in labels.items()}
                decoded_img = label_array(decoded_img, switched_labels)
                decoded_img = np.reshape(decoded_img, shape)
                decoded_img = Image.fromarray(decoded_img)
                decoded_img.save(output_path)

        print("\nDecompression is done.")

def main():
    input_folder = 'data/Mask'
    decoder_input_file = 'data/output.csv'

    encoder = Encoder(input_folder = input_folder,
                      output_csv = decoder_input_file)
    
    decoder = Decoder(input_file = decoder_input_file,
                      output_folder = 'data/reconstructed_masks')
           
if __name__ == "__main__":
    main()