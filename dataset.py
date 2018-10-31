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

    def __init__(self, debug=False, skip_rows=False):
        """Load the data from disk.

        Args:
            debug (bool): An option to choose whether to load all
              data.  If 'debug' is true, program will only read 1000 rows
              data from the csv file.
              However, one thing to pay attention is that if you load
              less data, the shape of DF is wrong, because some
              columns don't have any data until you read many many
              rows.
            skip_rows (bool): An option to load an evenly distributed
              sample of the dataset. If 'debug' is true, _approximately_
              1000 rows will be read from the csv file, but taken every
              _NUM_SKIP_ROWS_TRAIN and _NUM_SKIP_ROWS_TEST rows instead
              of just the first 1000 rows.

        """
        if skip_rows and not debug:
            raise ValueError('debug mode must be on to skip rows')
        rows_to_skip_train = 1
        rows_to_skip_test = 1
        
        if debug and not skip_rows:
            nrows = _NUM_ROWS_DEBUG
        else:
            nrows = None
        if skip_rows:
            rows_to_skip_train = _NUM_ROWS_TRAIN // _NUM_ROWS_DEBUG
            rows_to_skip_test = _NUM_ROWS_TEST // _NUM_ROWS_DEBUG
            
        type_change_columns = {"fullVisitorId": str,
                               "sessionId": str,
                               "visitId": str}
        json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
        date_columns = ['date', 'visitStartTime']

        converters = {column: self._make_json_converter(column)
                      for column in json_columns}

        self.train = pd.read_csv(os.path.join(_DATA_DIR, _TRAIN),
                                 converters=converters,
                                 dtype=type_change_columns,
                                 nrows=nrows, 
                                 skiprows=lambda i: i % rows_to_skip_train !=0)
        self.test = pd.read_csv(os.path.join(_DATA_DIR, _TEST),
                                converters=converters,
                                dtype=type_change_columns,
                                nrows=nrows, 
                                skiprows=lambda i: i % rows_to_skip_test !=0)

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
        df['encoding_medium'], df['encoding_referralPath'], df['encoding_source'] = self._make_traffic_source_preprocessing()
        df['encoding_campaign'], df['encoding_isTrueDirect'], df['encoding_keyword'] = self._another_traffic_source_preprocessing()
        df = pd.concat([df, self._make_browser_preprocessing()],axis=1,sort=True)
        
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

    def _make_traffic_source_preprocessing(self):
        """Create the encoding columns of trafficSource.medium,trafficSource.referralPath, trafficSource.source.

        Returns:
           A DataFrame containing three columns, encoding_medium, encoding_referralPath, encoding_source, for the
           training set.
        """
        # Get the trafficSource.medium,trafficSource.referralPath, trafficSource.source.
        train_df = self.train.copy(deep=False)
        le = preprocessing.LabelEncoder()
        to_encode = ['medium', 'referralPath', 'source']
        for item in to_encode:
            item_key = 'trafficSource.' + item
            encoding_key = 'encoding_' + item
            train_df[item_key] = train_df[item_key].fillna("missing")
            fitting_label = train_df[item_key].unique()
            le.fit(fitting_label)
            train_df[encoding_key] = le.transform(train_df[item_key])
        train_gdf = train_df.groupby('fullVisitorId')
        return train_gdf['encoding_medium'].sum(), train_gdf['encoding_referralPath'].sum(), train_gdf['encoding_source'].sum()

    def _another_traffic_source_preprocessing(self):
        """Create the encoding columns of trafficSource.campaign,trafficSource.isTrueDirect, trafficSource.keyword.

        Returns:
           A DataFrame containing three columns, encoding_campaign, encoding_isTrueDirect, encoding_keyword, for the
           training set.
        """
        # For 'campaign' & 'keyword'
        train_df = self.train.copy(deep=False)
        le = preprocessing.LabelEncoder()
        to_encode = ['campaign', 'keyword']
        for item in to_encode:
            item_key = 'trafficSource.' + item
            encoding_key = 'encoding_' + item
            train_df[item_key] = train_df[item_key].fillna("missing")
            fitting_label = train_df[item_key].unique()
            le.fit(fitting_label)
            train_df[encoding_key] = le.transform(train_df[item_key])
        # Now for 'isTrueDirect'
        item_key = 'trafficSource.isTrueDirect'
        encoding_key = 'encoding_isTrueDirect'
        train_df[encoding_key] = train_df[item_key].fillna(False)

        train_gdf = train_df.groupby('fullVisitorId')
        return train_gdf['encoding_campaign'].sum(), train_gdf['encoding_isTrueDirect'].sum(), train_gdf['encoding_keyword'].sum()
    
        def _make_browser_preprocessing(self):
        """Creates the encoding columns of device.browser, device.browserSize, device.browserVersion

        Returns:
            A Dataframe containing one hot encoded columns for unique values of device.browser,
            device.browserSize, device.browserVersion

        """
        train_df = self.train.copy(deep=False)
        browser = self._one_hot('device.browser')
        browserSize = self._one_hot('device.browserSize')
        browserVersion = self._one_hot('device.browserVersion')

        return pd.concat([browser,browserSize,browserVersion],axis=1,sort=True)

    def _one_hot(self, key):
        """Creates one hot encodings for categorical variables

        Args:
            key (string): name of column in dataframe to one hot encode

        Returns:
            A Dataframe containing one hot encoding of categorical variable, grouped by
            visitor ID. Columns are unique values of variable.
        """
        train_df = self.train.copy(deep=False)
        one_hot = pd.get_dummies(train_df[key],drop_first=True)
        one_hot.columns=[key+'_'+col for col in one_hot.columns.values]
        train_df = pd.concat([train_df,one_hot],axis=1,sort=True)
        train_dfg = train_df.groupby('fullVisitorId')
        return train_dfg[one_hot.columns.values].sum()

    def _make_json_converter(self, column_name):

        """Helper function to interpret columns in PANDAS."""
        return lambda x: {column_name: json.loads(x)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on the Google Analytics Dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    args = parser.parse_args()

    # Make sure we can load the dataset
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
