import glob
import time
import os
import numpy as np
from PIL import Image
from collections import deque
import csv
import sys
from csv import DictWriter, DictReader


def get_image_paths(input_folder: str, suffix: str = '.tif') -> list[str]:
    """
    Retrieve image file paths from the specified folder.

    Args:
        input_folder (str): The path to the folder containing the images.
        suffix (str): The file extension to filter images (default: '.tif').

    Returns:
        list[str]: A list of file paths that match the specified suffix.
    """
    return glob.glob(os.path.join(input_folder, f"*{suffix}"))


def read_image_as_array(path: str) -> np.ndarray:
    """
    Read an image file and convert it to a grayscale. Return as NumPy array.

    Args:
        path (str): The file path of the image to be read.

    Returns:
        np.ndarray: A NumPy array representation of the grayscale image.
    """
    return np.array(Image.open(path).convert('L'))


def is_binary(image: np.ndarray) -> bool:
    """
    Check if a given image is binary.

    Args:
        image (np.ndarray): The image represented as a NumPy array.

    Returns:
        bool: True if the image contains exactly two unique values, False otherwise.
    """
    return len(np.unique(image)) == 2


def create_label_dictionary(array: np.ndarray, method: str = 'standard') -> str:
    """
    Create a dictionary mapping unique values in an array to labels.

    Args:
        array (np.ndarray): The input array containing unique values.
        method (str, optional): The method to determine label assignment. 
            'standard' maps the first unique value to 'b' and the second to 'w'.
            'inverse' maps the first unique value to 'w' and the second to 'b'. 
            Defaults to 'standard'.

    Returns:
        dict: A dictionary mapping unique values to their corresponding labels.

    Raises:
        ValueError: If the method is not 'standard' or 'inverse'.
    """
    match method:
        case 'standard':
            labels = {
                np.sort(np.unique(array))[0] : 'b',
                np.sort(np.unique(array))[1] : 'w',
            }
            return labels

        case 'inverse':
            labels = {
                np.sort(np.unique(array))[0] : 'w',
                np.sort(np.unique(array))[1] : 'b',
            }
            return labels

        case _:
            raise ValueError("Method must be 'standard' or 'inverse'.")


def label_array(array : np.ndarray, labels: dict) -> np.ndarray:
    """
    Label elements in an array based on a provided dictionary.

    Args:
        array (np.ndarray): The input array to be labeled.
        labels (dict): A dictionary mapping values to their corresponding labels.

    Returns:
        np.ndarray: An array with the same shape as the input, with elements replaced 
        by their corresponding labels from the dictionary.
    """
    return np.vectorize(labels.get)(array)


def make_substring(counter: int, char: str, threshold: int = 2) -> str:
    """
    Helper function to create a substring based on the character count and a threshold.

    Args:
        counter (int): The number of occurrences of the character.
        char (str): The character to be repeated or suffixed.
        threshold (int, optional): The threshold for deciding the format of the 
        substring. Defaults is 2.

    Returns:
        str: A substring representing the character repeated `counter` times, or 
        formatted as '<count><char>' if `counter` exceeds the threshold.
    """
    if counter <= threshold:
        substring = counter * char
    else:
        substring = f"{counter}{char}"
    
    return substring


def rle_encoding(array: np.ndarray) -> str:
    """
    Encode a 1D numpy array using Run-Length Encoding (RLE).

    Args:
        array (np.ndarray): A 1D numpy array containing the data to be encoded.

    Returns:
        str: The RLE representation of the input array as a string.
    """
    
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
    """
    Decode a Run-Length Encoded (RLE) string back into a numpy array.

    Args:
        input (str): The RLE encoded string to be decoded.

    Returns:
        np.ndarray: A 1D numpy array representing the decoded data.
    """

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
    """
    Check if the given file path has a CSV file extension.

    Args:
        path (str): The file path to check.

    Returns:
        bool: True if the file has a '.csv' extension, otherwise raises a ValueError.
    
    Raises:
        ValueError: If the input file is not a '.csv' file.
    """
    
    _, extension = os.path.splitext(path)

    if extension == '.csv':
        return True
    else:
        raise ValueError("Input file is not '.csv' file.")


class Encoder():
    """
    Encoder class for RLE encoding of images in a specified folder.

    This class automatically collects image paths, performs RLE encoding on binary images,
    and writes the results to a CSV file.

    Attributes:
        input_folder (str): Path to the folder containing input images.
        output_csv (str): Path to the output CSV file (default is './output.csv').
        image_suffix (str): Suffix for image files to be processed (default is '.tif').
    """
    def __init__(self,
                 input_folder: str,
                 output_csv: str = './output.csv',
                 image_suffix: str = '.tif',
                 log_excluded = True,
                 labels_method = 'standard',
                 add_ratio = False,
                 ):
        """
        Initializes the Encoder class.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_csv (str): Path to the output CSV file (default is './output.csv').
            image_suffix (str): Suffix for image files to be processed (default is '.tif').
            log_excluded (bool): If True, logs excluded files to 'excluded.txt' (default is True).
            labels_method (str): Method for creating labels ('standard' or 'inverse', default is 'standard').
        """

        self.input_folder = input_folder
        self.output_csv = output_csv
        self.image_suffix = image_suffix
        self.log_excluded = log_excluded
        self.labels_method = labels_method
        self.add_ratio = add_ratio

        # Start encoding automatically 
        self.forward()

    def forward(self,):
        """
        Performs the encoding process.

        Collects image paths, encodes binary images using RLE, and writes the results to
        a CSV file.

        """
        is_csv(self.output_csv)
        log_excluded = self.log_excluded,
        labels_method = self.labels_method

        print("Collect image paths...")
        paths = get_image_paths(self.input_folder, self.image_suffix)
        excluded_files = []

        print(f"Create {self.output_csv} file ...")
        with open(self.output_csv, 'w') as file:
            if self.add_ratio:
                fieldnames = ['Image', 'Shape', 'Labels', 'RLE', 'Ratio']
            else:
                fieldnames = ['Image', 'Shape', 'Labels', 'RLE']
            writer = DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            print("Start RLE encoding ...")
            for index, path in enumerate(paths):

                print(f"Encoding {index + 1}th image / {len(paths)} images.", end = '\r')
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
                    if self.add_ratio:
                        ratio = len(rle) / len(img)
                        writer.writerow({
                            'Image' : name,
                            'Shape' : shape,
                            'Labels' : labels,
                            'RLE' : rle,
                            'Ratio' : ratio
                        })
                    
                    else:
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
    """
    Decoder class for reconstructing images from RLE encoded data in a CSV file.

    This class automatically reads a specified CSV file, decodes RLE encoded images, 
    and saves the reconstructed images to a specified output folder.

    Attributes:
        input_file (str): Path to the input CSV file containing RLE encoded data.
        output_folder (str): Path to the folder where reconstructed images will be saved (default is './reconstructed_masks').
        image_suffix (str): Suffix for the output image files (default is '.tif').
    """

    def __init__(self,
                 input_file: str,
                 output_folder: str = './reconstructed_masks',
                 image_suffix: str = '.tif'
                 ):
        
        """
        Initializes the Decoder class.

        Args:
            input_file (str): Path to the input CSV file containing RLE encoded data.
            output_folder (str): Path to the folder where reconstructed images will be 
            saved (default is './reconstructed_masks').
            image_suffix (str): Suffix for the output image files (default is '.tif').
        """

        self.input_file = input_file
        self.output_folder = output_folder
        self.image_suffix = image_suffix

        self.forward()


    def forward(self,):
        """Performs the decoding process.

        Checks the input file, creates the output directory, loads the RLE data from the
        CSV file, and reconstructs images by decoding the RLE encoded data.

        Raises:
            ValueError: If the input file is not a valid CSV file.
        """

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

        # Reader need to reinitalise after count rows for some reason.
        with open(self.input_file, 'r') as file:       
            reader = DictReader(file)

            print("Start RLE decoding ...")
            for index, row in enumerate(reader):
                print(f"Deecoding {index + 1}th image / {rows} images.", end = '\r')
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

        print("\nDecompression is done.\n")


def check_reconstruction_is_valid(original_images: str,
                                  reconstructed_images: str,
                                  suffix: str = '.tif') -> bool:
    
    print("Check reconstruction validity ...")
    original_paths = get_image_paths(original_images, suffix)
    for index, path in enumerate(original_paths):
        print(f"Scan {index+1}th / {len(original_paths)} paths.", end = "\r")
        name = os.path.basename(path)
        try:
            reconst_img = read_image_as_array(os.path.join(reconstructed_images,
                                                           name))
            original = read_image_as_array(path)

            if not np.array_equal(original, reconst_img):
                return print(f"\nReconstruction is invalid on path: \n{path}.")
            
        except FileNotFoundError:
            continue

    return print("\nReconstruction is valid.")

def main():

    # Parameters
    input_folder = "/Users/fazekasbalint/Documents/Programing/2024/FB_Major/data/images/patches_for_Labkit_512/masks/q1"
    renconstructed_images = "data/reconstructed_masks"
    decoder_input_file = 'data/output.csv'


    # Encoder block
    start_time = time.time()
    encoder = Encoder(input_folder = input_folder,
                      output_csv = decoder_input_file,
                      add_ratio = True)
    end_time = time.time()
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    print(f"Encoding execution time: {int(minutes)} minutes and {seconds:.2f} seconds.\n")
    

    # Decoder block
    start_time = time.time()
    decoder = Decoder(input_file = decoder_input_file,
                      output_folder = 'data/reconstructed_masks')
    end_time = time.time()
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    print(f"Decoding execution time: {int(minutes)} minutes and {seconds:.2f} seconds.\n")


    # Check validity
    check_reconstruction_is_valid(input_folder,
                                  renconstructed_images)


if __name__ == "__main__":
    main()