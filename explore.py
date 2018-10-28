from __future__ import print_function

import matplotlib.pyplot as plt

import dataset
import explore_utils
import argparse


def main(args):

    print("Loading the dataset...")
    data = dataset.Dataset(args.debug, args.skip_rows)

    print("Number of rows in the training set:    ", len(data.train))
    print("Number of columns in the training set: ", len(data.train.columns))

    print("Number of rows in the test set:        ", len(data.test))
    print("Number of columns in the test set:     ", len(data.test.columns))

    # Number of visits

    print("The most visit times for a customer in train set is: ",
          explore_utils.find_most_visit(data))

    # Customer spending percentiles

    percentiles = [95, 97.5, 99, 99.9, 99.99]
    percentile_values = explore_utils.find_customer_revenue_percentiles(
        data,
        percentiles)
    for p, pv in zip(percentiles, percentile_values):
        print("%2.2f%% of customers spend less than: $%.2f" % (p, pv))

    # Total average revenue per trafficSource 
    print("The revenue per trafficSource is: ")
    print(explore_utils.revenue_per_trafficsource(data))

    # The fraction of transactions that have non-zero revenue
    print("Fraction of transactions that have non-zero revenue:    ",
          explore_utils.find_fraction_of_transactions_with_non_zero_revenue(data))

    # Most common Sources of Traffic and counts
    num_of_sources = 6
    most_common_sources = explore_utils.find_most_common_traffic_sources(data, num_of_sources)
    print("\nThe {} most common sources of traffic are :".format(num_of_sources))
    for source in most_common_sources.index:
        print(' {}: {}'.format(source, most_common_sources[source]))

    # channelGroupings and customer revenue (#31)
    counts, means = explore_utils.find_channel_grouping_revenue(data)
    print('Channel Grouping Counts:')
    print(counts)

    print('Mean Total Revenue by Channel Grouping')
    print(means)

    # transactionRevenue by region

    trans_by_region = explore_utils.find_transaction_by_region(data)
    print("The transaction revenues by region are: ", trans_by_region)


    # Unique Visitor Percentage
    print("%2.2f%% of all visitors to the site visit exactly once." % explore_utils.find_one_visit_percent(data))

    # Statistics of sales made by first time visitors vs returning visitors
    first_and_return_visits = explore_utils.find_return_visit_stats(data)
    first_and_return_visits = first_and_return_visits.round(2)
    print("\nStatistics of total transactions for unique visitors: \n{}\n"
        .format(first_and_return_visits))

    # Summary statistics for revenue generated, by device
    # First print-out includes sessions that didn't produce revenue
    # Second print-out only includes sessions that generated revenue
    print("Revenue summary statistics by device, zeroes included: \n",
          explore_utils.find_revenue_summary_statistics_for_devices(data, True))
    print ("Revenue summary statistics by device, zeroes excluded: \n",
           explore_utils.find_revenue_summary_statistics_for_devices(data, False))

    # Prints out what percent of revenue generating sessions were accessed via a particular device
    print("Percent of revenue generating sessions that used a particular device: \n",
          explore_utils.find_percent_sessionIds_using_certain_device(data))

    # Prints out the percent of total revenue that can be attributed to sessions
    # accessed via a particular device
    print("Percent of total revenue attributed to sessions using a particular device: \n",
          explore_utils.find_percent_of_total_revenue_by_device(data))

    # Prints out the percent of sessions accessed via a particular device that
    # generated revenue
    print("Percent of total sessions using a particular device that generated revenue: \n",
          explore_utils.find_percent_device_uses_generating_revenue(data))

    # Plots (1) monthly and (2) yearly histograms of number of site visits,
    # (3) total revenue scattergram and (4) percent visitors who make purchases
    # scattergrams
    dates, total_revenue_per_day, daily_percent = explore_utils.find_seasonal_trends(data)
    fix, axs = plt.subplots(4, 1, sharex=True)
    axs[3].set_xlabel('Date')

    axs[0].hist(dates, bins=12)
    axs[0].set_title('Monthly')
    axs[0].set_ylabel('Number of visits')

    axs[1].hist(dates, bins=54)
    axs[1].set_title('Yearly')
    axs[1].set_ylabel('Number of visits')

    axs[2].scatter(total_revenue_per_day.index, total_revenue_per_day)
    axs[2].set_title('Revenue')
    axs[2].set_ylabel('Total revenue')

    axs[3].scatter(daily_percent.index, daily_percent)
    axs[3].set_title('Percent purchase')
    axs[3].set_ylabel('Percent')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Explore the Google Analytics dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    parser.add_argument('--skip-rows', dest='skip_rows', action='store_true',
                        help='run with skip_rows enabled')
    args = parser.parse_args()

    main(args)
