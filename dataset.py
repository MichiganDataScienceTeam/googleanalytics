from __future__ import print_function

import os.path
import pandas as pd
import numpy as np
import json
import argparse
from sklearn import preprocessing

_DATA_DIR = './data'
_TRAIN = 'train.csv'
_TEST = 'test.csv'

_NUM_ROWS_TRAIN = 903653
_NUM_ROWS_TEST = 804684

_NUM_ROWS_DEBUG = 1000


class Dataset():
    """The Google Analytics dataset."""

    def __init__(self, debug=False):
        """Load the data from disk.

        Args:
            debug (bool): An option to choose whether to load all
              data.  If 'debug' is true, program will only read 100 rows
              data from the csv file.
              However, one thing to pay attention is that if you load
              less data, the shape of DF is wrong, because some
              columns daon't have any data until you read many many
              rows.

        """

        if(debug):
            nrows = _NUM_ROWS_DEBUG
        else:
            nrows = None
        type_change_columns = {"fullVisitorId": str,
                               "sessionId": str,
                               "visitId": str}
        json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']

        converters = {column: self._make_json_converter(column)
                      for column in json_columns}

        self.train = pd.read_csv(os.path.join(_DATA_DIR, _TRAIN),
                                 converters=converters,
                                 dtype=type_change_columns,
                                 nrows=nrows)
        self.test = pd.read_csv(os.path.join(_DATA_DIR, _TEST),
                                converters=converters,
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

    def preprocess(self, do_val_split=False):
        """Preprocess the dataset.

        Args:
           do_val_split (bool): Whether to do a validation split. Not
              yet implemented.

        Returns:
           A preprocessed version of the training set with only
           numerical data for ML models.
        """

        if do_val_split:
            raise NotImplementedError(
                'Validation split not yet implemented.')

        df = pd.DataFrame({'visitorId': self.train['fullVisitorId'].unique()})
        df.set_index('visitorId', inplace=True)

        # Preprocessing operations go here.
        df['log_sum_revenue'] = self._make_log_sum_revenue()
        df['geoNetwork_networkDomain'] = self._convert_geoNetwork_domain()
        
        return df

    def _make_log_sum_revenue(self):
        """Create the log_sum_revenue column.

        Returns:
           A DataFrame containing one column, log_sum_revenue, for the
           training set.
        """

        # Get revenue and fill NaN with zero
        train_df = self.train.copy(deep=False)
        train_df['revenue'] = train_df['totals.transactionRevenue']
        train_df['revenue'] = train_df['revenue'].astype('float').fillna(0)

        # Group by visitor and sum, log
        train_gdf = train_df.groupby('fullVisitorId')
        train_revenue_sum = train_gdf['revenue'].sum()
        train_revenue_log_sum = (train_revenue_sum + 1).apply(np.log)
        
        return train_revenue_log_sum

    def _make_json_converter(self, column_name):

        """Helper function to interpret columns in PANDAS."""
        return lambda x: {column_name: json.loads(x)}

    def _convert_geoNetwork_domain(self):
        """Ont hot encode domain, location, region, subContinent in geoNetwork.
        Missing value automatically imputed by one hot encoder.
        Standardize using normal distribution.
        Group by fullVisitorID.

        Returns:
            A DataFrame containing preprocessed geoNetwork Data with one hot encoding.
        """
        train_df = self.train.copy(deep=False)
        to_encode = ['networkDomain', 'networkLocation', 'region', 'subContinent']
        results = pd.DataFrame()
        for index, row in train_df.iterrows():
            for item in to_encode:
                individual_key = 'geoNetwork.' + item
                row[individual_key] = row['geoNetwork']['geoNetwork'][item]
        for item in to_encode:
            individual_key = 'geoNetwork.' + item
            encoded = pd.get_dummies(train_df[individual_key], prefix=individual_key)
            results = pd.concat([results, encoded], axis=1)
        columns = results.columns
        results[columns] = preprocessing.scale(results[columns])
        results[columns] = preprocessing.normalize(results[columns], norm='l2')

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on the Google Analytics Dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    args = parser.parse_args()

    # Make sure we can load the dataset
    args.debug = True
    dataset = Dataset(debug=args.debug)

    # Sanity check, make sure we have the right number of rows
    num_train = len(dataset.train)
    num_test = len(dataset.test)
    if args.debug:
        assert num_train == _NUM_ROWS_DEBUG
        assert num_test == _NUM_ROWS_DEBUG
    else:
        assert num_train == _NUM_ROWS_TRAIN, 'Incorrect number of training examples found.'
        assert num_test == _NUM_ROWS_TEST, 'Incorrect number of test examples found.'

    print('Successfully loaded the dataset.')
    dataset.preprocess()
