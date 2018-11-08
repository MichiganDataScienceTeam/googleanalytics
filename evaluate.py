from __future__ import print_function

import os
import argparse

import pandas as pd
import numpy as np


def main(args):
    """Evaluate the Log-RMSE of a set of predictions.

    Log-RMSE = sqrt( 1 / N * sum( (log (1 + y))^2 - (log (1 + y'))^2 ) )

    Basically, we convert the ground truth y and predictions y' to log(1 + y),
    log(1 + y'). Then we compute RMSE of these values.

    Both the ground truth and prediction file must have a format like this:

    fullVisitorId,transactionRevenue
    1234567890,0.0
    1234567891,0.0
    ...


    """

    print('Loading ground truth csv: ', args.ground_truth_file)
    gt = pd.read_csv(
        args.ground_truth_file,
        dtype=str).set_index('fullVisitorId')

    print('Loading prediction csv: ', args.prediction_file)
    pred = pd.read_csv(
        args.prediction_file,
        dtype=str).set_index('fullVisitorId')

    gt = pd.to_numeric(gt['transactionRevenue'])
    pred = pd.to_numeric(pred['transactionRevenue'])

    if args.skip_log:
        print('Computing RMSE without log.')
        result = rmse(gt, pred)
    else:
        print('Computing RMSE with log.')
        result = rmse_log(gt, pred)

    print('RMSE:', result)


def rmse_log(gt, preds):
    """Compute the RMSE in between two series log space."""
    return rmse(np.log(1+gt), np.log(1+preds))


def rmse(gt, preds):
    """Compute the RMSE between two series."""
    # TODO: check for case where indices don't align
    return np.sqrt(((gt - preds) * (gt - preds)).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the RMSE of a set of predictions in log space.')
    parser.add_argument(
        '--ground_truth_file', default='./processed_data/val_revenues.csv',
        help='Location where ground truth revenues are stored.',
        type=str)
    parser.add_argument(
        '--prediction_file', default='./processed_data/val_sample_predictions.csv',
        help='Location where predicted revenues are stored.',
        type=str)
    parser.add_argument(
        '--skip_log', action='store_true',
        help='Whether to skip conversion to log space. Useful if files are already converted.')
    args=parser.parse_args()

    main(args)
