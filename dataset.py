import os.path
import pandas as pd
import json
import explore_utils

_DATA_DIR = './data'
_TRAIN = 'train.csv'
_TEST = 'test.csv'


class Dataset():
    """The Google Analytics dataset."""

    def __init__(self, debug=False):
        """Load the data from disk.

        Args:
            debug (bool): An option to choose whether to load all
              data.  If 'debug' is true, program will only read 100 rows
              data from the csv file. 
              However, one thing to pay attention is that if you load less data,
              the shape of DF is wrong, because some columns daon't have any
              data until you read many many rows.
        """

        if(debug):
            nrows = 100
        else:
            nrows = None
        type_change_columns = {"fullVisitorId": str,
                               "sessionId": str,
                               "visitId": str}
        json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
        self.train = pd.read_csv(os.path.join(_DATA_DIR, _TRAIN),
                                 converters={
                                     column: json.loads
                                     for column in json_columns},
                                 dtype=type_change_columns,
                                 nrows=nrows)

        self.test = pd.read_csv(os.path.join(_DATA_DIR, _TEST),
                                converters={
                                    column: json.loads
                                    for column in json_columns},
                                dtype=type_change_columns,
                                nrows=nrows)
        for column in json_columns:
            train_column_as_df = pd.io.json.json_normalize(self.train[column])
            test_column_as_df = pd.io.json.json_normalize(self.test[column])
            self.train = self.train.merge(train_column_as_df,
                                          right_index=True,
                                          left_index=True)
            self.test = self.test.merge(test_column_as_df,
                                        right_index=True,
                                        left_index=True)


if __name__ == '__main__':

    # Make sure we can load the dataset
    dataset = Dataset()

    # Sanity check, make sure we have the right number of rows
    num_train = len(dataset.train)
    num_test = len(dataset.test)
    assert num_train == 903653, 'Incorrect number of training examples.'  # 903653
    assert num_test == 804684, 'Incorrect number of test examples.'  # 804684
    
    print('Successfully loaded the dataset.')