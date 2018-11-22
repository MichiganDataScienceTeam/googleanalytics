from __future__ import print_function

import os
import argparse
import json

import pandas as pd
import numpy as np
import random

# August 1 - October 15
_TRAIN_START_DATE = np.datetime64('2016-06-01')
_TRAIN_END_DATE = np.datetime64('2016-10-16')
# Temporarily moved back a year while we wait for data from Kaggle
#_TRAIN_END_DATE = np.datetime64('2017-10-16')

# December 1 - January 31
# Temporarily moved back a year while we wait for data from Kaggle
#_LABEL_START_DATE = np.datetime64('2017-12-01')
#_LABEL_END_DATE = np.datetime64('2018-02-01')
_LABEL_START_DATE = np.datetime64('2016-12-01')
_LABEL_END_DATE = np.datetime64('2017-02-01')

# August 1 - October 15 2018
_TEST_START_DATE = np.datetime64('2018-05-01')
_TEST_END_DATE = np.datetime64('2018-10-16')

_TRAIN_FRAC = 0.80

def main(args):
    """Split the training validation data into two files.

    We choose a validation set intended to match the characteristics of the
    test set. Specifically, the features will come from the time period leading
    up to the holiday season, and the labels will come from the holiday season.
    The validation split will be made over users, such that a percentage of the
    users who appear in the feature time period will be segmented off into the
    validation set.

    Features: August 1st 2016 - October 15 2017
    Labels: December 1 2017 - January 31 2018

    For reference, the competition data spans these time periods:

    train.csv:    August 1st 2016 - August 1st 2017
    test.csv:     August 2nd 2017 - April 30th 2018 [no labels]

    To be released by Nov 9:

    train_v2.csv: May 1st 2018 - April 30th 2018
    test_v2.csv:  May 1st 2018 - October 15th 2018 [not sure if labels]

    For the training period, we include all sessions that occured during this
    period. For the validation set, we include only the sessions that involve
    users that appeared during the training period.

    """
    
    print('Loading train csv...')
    train = pd.read_csv(
        os.path.join(args.data_dir, args.input_train_file),
        dtype=str)

    print('Loading test csv...')
    test = pd.read_csv(
        os.path.join(args.data_dir, args.input_test_file),
        dtype=str)

    df = pd.concat((train, test), axis=0).reset_index(drop=True)
    dates = pd.to_datetime(df['date'], format='%Y%m%d')

    print('Splitting by date...')
    feat_idx = (_TRAIN_START_DATE <= dates) & (dates < _TRAIN_END_DATE)
    lab_idx = (_LABEL_START_DATE <= dates) & (dates < _LABEL_END_DATE)
    test_idx = (_TEST_START_DATE <= dates) & (dates < _TEST_END_DATE)

    df_feat = df[feat_idx]
    df_label = df[lab_idx]
    df_test = df[test_idx]

    print('Splitting by customer id...')
    visitors = np.unique(df_feat['fullVisitorId'])
    visitors_test = np.unique(df_test['fullVisitorId'])

    print('Unique visitors in train set: ', len(visitors))
    print('Unique visitors in test set: ', len(visitors_test))

    # p is the approximation of percent of dataset to split
    p = args.percent_split/100
    print('Splitting percent of users: ', p)

    if args.fresh_split:
        print('Choosing new train/val split...')

        visitors = np.random.choice(visitors,int(len(visitors)*p),False)

        n_visitors = len(visitors)
        n_train = int(n_visitors * _TRAIN_FRAC)
        n_valid = n_visitors - n_train

        np.random.shuffle(visitors)

        visitors_df = pd.DataFrame({
            'fullVisitorId': visitors,
            'split': ['train']*n_train + ['valid']*n_valid
        })

        print('Writing the split to csv...')
        visitors_df.to_csv(
            os.path.join(args.data_dir, args.split_file),
            index=False
        )

    else:
        # p is the approximation of percent of dataset to split
        p = args.percent_split/100
        print('Loading split from csv...')
        visitors_df = pd.read_csv(
            os.path.join(args.data_dir, args.split_file),
            dtype=str,
            skiprows=lambda i: i>0 and random.random() > p
        )

    visitors_df = visitors_df.set_index('fullVisitorId')
    visitors_train = visitors_df.index[visitors_df['split']=='train']
    visitors_valid = visitors_df.index[visitors_df['split']=='valid']

    print('Number of visitors in train minus val set:', len(visitors_train))
    print('Number of visitors in val set:', len(visitors_valid))

    df_trainminusval = df_feat[df_feat['fullVisitorId'].isin(visitors_train)]
    df_val = df_feat[df_feat['fullVisitorId'].isin(visitors_valid)]

    print('Making labels...')
    df_label = df_label.set_index('fullVisitorId')
    df_label = df_label['totals'].apply(json.loads)
    df_totals = pd.io.json.json_normalize(df_label)
    df_totals['fullVisitorId'] = df_label.index
    df_totals = df_totals.set_index('fullVisitorId')

    if 'transactionRevenue' not in df_totals.columns:
        print('Warning: no transaction revenue found!')
        df_revenue = pd.Series()
    else:
        df_revenue = df_totals['transactionRevenue'].astype('float')

    df_revenue.fillna(0, inplace=True)
    gdf_revenue = df_revenue.groupby(df_revenue.index)
    sum_revenue = gdf_revenue.sum()

    sum_revenue_trainminusval = sum_revenue.reindex(visitors_train, fill_value=0)
    sum_revenue_val = sum_revenue.reindex(visitors_valid, fill_value=0)

    sum_revenue_trainminusval = sum_revenue_trainminusval.reset_index().rename(
        {'index': 'sumRevenue'})
    sum_revenue_val = sum_revenue_val.reset_index().rename(
        {'index': 'sumRevenue'})
    all_zeros_val = sum_revenue_val.copy()
    all_zeros_val['transactionRevenue'] = 0

    print('Saving...')
    df_trainminusval.to_csv(
        os.path.join(args.output_dir, args.output_train_file),
        index=False)
    df_val.to_csv(
        os.path.join(args.output_dir, args.output_valid_file),
        index=False)
    df_test.to_csv(
        os.path.join(args.output_dir, args.output_test_file),
        index=False)
    sum_revenue_trainminusval.to_csv(
        os.path.join(args.output_dir, args.output_train_label_file),
        index=False)
    sum_revenue_val.to_csv(
        os.path.join(args.output_dir, args.output_valid_label_file),
        index=False)
    all_zeros_val.to_csv(
        os.path.join(args.output_dir, args.output_valid_sample_predictions_file),
        index=False)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split the training and validation data into two files.')
    parser.add_argument(
        '--data_dir', default='./data',
        help='Location where data is stored.',
        type=str)
    parser.add_argument(
        '--input_train_file', default='train.csv',
        help='Train file to be split.',
        type=str)
    parser.add_argument(
        '--input_test_file', default='test.csv',
        help='Test file to be split.',
        type=str)
    parser.add_argument(
        '--output_dir', default='./processed_data',
        help='Location where data is written.',
        type=str)
    parser.add_argument(
        '--output_train_file', default='trainminusval_visits.csv',
        help='Destination file for new training set.',
        type=str)
    parser.add_argument(
        '--output_train_label_file', default='trainminusval_revenues.csv',
        help='Destination file for new training set.',
        type=str)
    parser.add_argument(
        '--output_valid_file', default='val_visits.csv',
        help='Destination file for new validation set.',
        type=str)
    parser.add_argument(
        '--output_valid_label_file', default='val_revenues.csv',
        help='Destination file for new validation set.',
        type=str)
    parser.add_argument(
        '--output_valid_sample_predictions_file', default='val_sample_predictions.csv',
        help='Destination file for new validation set.',
        type=str)
    parser.add_argument(
        '--output_test_file', default='test_visits.csv',
        help='Destination file for new test set.',
        type=str)
    parser.add_argument(
        '--split_file', default='split_ids.csv',
        help='Table of ids and corresponding split.',
        type=str)
    parser.add_argument(
        '--fresh_split', action='store_true',
        help='Whether to create a new split.')
    parser.add_argument(
        '--percent_split', default=100,
        help='Percent of users to include in split csv (0,100]',
        type=int
    )
    args=parser.parse_args()

    main(args)
