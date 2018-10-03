from __future__ import print_function

import dataset
import explore_utils
import argparse


def main(args):

    print("Loading the dataset...")
    data = dataset.Dataset(args.debug)

    print("Number of rows in the training set:    ", len(data.train))
    print("Number of columns in the training set: ", len(data.train.columns))

    print("Number of rows in the test set:        ", len(data.test))
    print("Number of columns in the test set:     ", len(data.test.columns))

    # Number of visits

    print("The most visit times for a customer in train set is: ",
          explore_utils.find_most_visit(data))

    #   Mean number of visits for socially engaged customers

    print("The mean visit times for socially engaged customers is: ",
          explore_utils.social_active_customers_revenue(data))

    # Customer spending percentiles

    percentiles = [95, 97.5, 99, 99.9, 99.99]
    percentile_values = explore_utils.find_customer_revenue_percentiles(
        data,
        percentiles)
    for p, pv in zip(percentiles, percentile_values):
        print("%2.2f%% of customers spend less than: $%.2f" % (p, pv))

    # Most common Sources of Traffic and counts
    num_of_sources = 6
    most_common_sources = explore_utils.find_most_common_traffic_sources(data,num_of_sources)
    print("\nThe {} most common sources of traffic are :".format(num_of_sources))
    for source in most_common_sources.index:
        print(' {}: {}'.format(source,most_common_sources[source]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Explore the Google Analytics dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    args = parser.parse_args()

    main(args)
