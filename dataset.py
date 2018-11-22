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
        df['geoNetwork.first_x_longitude'], df['geoNetwork.last_x_longitude'], df['geoNetwork.first_y_longitude'], df['geoNetwork.last_y_longitude'] = self._preprocess_longitudes_and_latitudes()
        
        # One-hot encode the countries.
        new_country_col = pd.Series(df['geoNetwork.country'])
        country_dummies_df = pd.get_dummies(new_country_col)
        
        # One-hot encode the metropolitan areas.
        new_metro_col = pd.Series(df['geoNetwork.metro'])
        metro_dummies_df = pd.get_dummies(new_metro_col)
        
        # Merge the dummy dataframes with the main training dataframe.
        df = df.join(country_dummies_df)
        df = df.join(metro_dummies_df)
        
        return df
    
    def _preprocess_longitudes_and_latitudes(self):
        """Preprocesses the columns u'geoNetwork.latitude',
           u'geoNetwork.longitude', and u'geoNetwork.metro' in the dataset train.csv.
           Creates columns of the first and last x and y longitudes and latitudes (4 columns) of each visitor.
           Also creates dummy columns of the country and metropolitan area of each visit.
           
           Returns:
               The training dataframe (indexed by visitor) with the added longitude/latitude columns.
        """
        
        train_df = self.train.copy(deep=False)
        
        # Preprocess the numeric columns. Group by visitor and standardize, impute missing values, and normalize
        
        # Impute the missing values in latitudes and longitudes.
        imp = SimpleImputer(missing_values='not available in demo dataset', strategy='mean')
        train_df['imputed_latitude'] = imp.transform(train_df['geoNetwork.latitude'])
        train_df['imputed_longitude'] = imp.transform(train_df['geoNetwork.longitude'])
        
        # Convert longitude and latitude into x and y coordinates.
        # First convert from degrees to radians, then take sin and cosine.
        train_df['x_longitude'] = train_df['geoNetwork.longitude'] * (math.pi / 180)
        train_df['x_longitude'] = numpy.cos(train_df['x_longitude'])
        train_df['y_longitude'] = train_df['geoNetwork.longitude'] * (math.pi / 180)
        train_df['y_longitude'] = numpy.sin(train_df['y_longitude'])
        
        train_df['x_latitude'] = train_df['geoNetwork.latitude'] * (math.pi / 180)
        train_df['x_latitude'] = numpy.cos(train_df['x_latitude'])
        train_df['y_latitude'] = train_df['geoNetwork.latitude'] * (math.pi / 180)
        train_df['y_latitude'] = numpy.sin(train_df['y_latitude'])
        
        # Sort by date.
        train_df = train_df.sort_values(by = ['date'])
        
        # Goup by Visitor ID.
        train_gdf = train_df.groupby('fullVisitorId')
        
        # First, use the train_df data to convert long & lat from degrees to radians, then take sin and cosine.
        # Then, group by fullVisitorID, and for each Visitor as a row, create a column with the Visitor's first
        # and last x and y longitudes and latitudes.
        
        # The first Longitudes of each visitor are in train_gdf['x_longitude'].first()
        # The last  Longitudes of each visitor are in train_gdf['x_longitude'].last()
        # The first Latitudes of each visitor are in train_gdf['x_latitude'].first()
        # The last  Latitudes of each visitor are in train_gdf['x_latitude'].last()
        df['geoNetwork.first_x_longitude'] = train_gdf['x_longitude'].first()
        df['geoNetwork.last_x_longitude'] = train_gdf['x_longitude'].last()
        df['geoNetwork.first_y_longitude'] = train_gdf['y_latitude'].first()
        df['geoNetwork.last_y_longitude'] = train_gdf['y_latitude'].last()
        
        return train_gdf['x_longitude'].first(), train_gdf['x_longitude'].last(), train_gdf['y_latitude'].first(), train_gdf['y_latitude'].last()
        
        
        

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
