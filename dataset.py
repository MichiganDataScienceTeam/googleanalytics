import os.path
import pandas as pd

_DATA_DIR = './data'
_TRAIN = 'train.csv'
_TEST = 'test.csv'


class Dataset():
    """The Google Analytics dataset."""

    def __init__(self):
        """Load the data from disk."""

        self.train = pd.read_csv(os.path.join(_DATA_DIR, _TRAIN))
        self.test = pd.read_csv(os.path.join(_DATA_DIR, _TEST))


if __name__ == '__main__':

    # Make sure we can load the dataset
    dataset = Dataset()

    # Sanity check, make sure we have the right number of rows
    num_train = len(dataset.train)
    num_test = len(dataset.test)
    assert num_train == 903653, 'Incorrect number of training examples.'
    assert num_test == 804684, 'Incorrect number of test examples.'

    print('Successfully loaded the dataset.')
