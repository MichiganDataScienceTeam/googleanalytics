import pandas as pd
import numpy as np
import pandas as pd

def find_most_visit(dataset):
    """Find what is the most visited times of single customer.(only in train set)

    args:
        dataset (Dataset): the google analytics dataset.

    returns:
            The most visited times in trainset.
    """

    train_df = dataset.train.copy()

    train_gdf = train_df.groupby("fullVisitorId")[
        'visitNumber'].sum().reset_index()
    max_visit = train_gdf['visitNumber'].max()

    return max_visit


def find_customer_revenue_percentiles(
        dataset,
        percentiles=[95, 99, 99.9, 99.99]):
    """Find percentiles of the per-customer revenue.

    args:
       dataset (Dataset): the google analytics dataset
       percentiles (list of floats): the percentiles to find

    returns:
       The values of the per-customer revenue at the given
       percentiles in the training set.

    """

    train_df = dataset.train.copy(deep=False)
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float)
    revenue_per_customer = train_df.groupby('fullVisitorId')['revenue']
    total_revenue_per_customer = revenue_per_customer.sum().fillna(0) / 10000.0

    values = [np.percentile(total_revenue_per_customer.values, percentile)
              for percentile in percentiles]

    return values


def find_most_common_traffic_sources(dataset, num=5):
    """ Find n most common traffic sources

    args:
       dataset (Dataset): the google analytics dataset
       num(optional): Number of most common traffic sources to return (by default 5)
    returns:
       Series of n (5 by default) most common traffic sources.

    """
    return dataset.train['trafficSource.source'].value_counts().head(num)

def find_one_visit_percent(dataset):
    """Finds the percent of visitors to the store that only visit once

    args:
        dataset (Dataset): the google analystics dataset

    returns:
        The percent of total customers that have only visited once, based on their ID

    """
    data = dataset.train.copy()

    #gets DataFrame of each visitor's total number of visits
    total_visits = data.groupby("fullVisitorId")['visitNumber'].sum()

    #counts all instances where the total visit number is exactly 1
    one_visit_count = np.sum(total_visits == 1)

    #divides by the total number of data points inspected
    percent_one_time_visitors = (100.*(one_visit_count))/(total_visits.size)

    #returns this percent
    return percent_one_time_visitors

def find_channel_grouping_revenue(dataset):
    """
    args:
       dataset (Dataset): the google analytics dataset

    returns:
       Tuple (dict, dict) containing mapping from channelGrouping name to count
       and mapping from channelGrouping name to average revenue in dollars.
    """

    train_df = dataset.train
    df = pd.DataFrame(train_df, columns=['channelGrouping', 'totals.transactionRevenue', 'fullVisitorId'])
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].fillna(0)
    groupings = df['channelGrouping'].unique()
    counts = {grouping: df[df['channelGrouping'] == grouping].shape[0] for grouping in groupings}
    means = {grouping: np.mean(df[df['channelGrouping'] == grouping]['totals.transactionRevenue'].astype('int64')) / 10000 for grouping in groupings}

    return counts, means

def find_transaction_by_region(data):
    """ Find the average transaction revenue by region

    args:
        dataset (Dataset): the google analytics Dataset

    returns:
        Dataframe of total transaction revenue by region in ascending order.

    """
    train_df = data.train
    data = train_df.copy(deep=False)
    data['rev'] = data['totals.transactionRevenue'].fillna(0).astype(float)
    avg = data.groupby('geoNetwork.region')['rev'].mean()
    avg = pd.DataFrame(avg)
    new_df = avg[avg['rev']>0]
    new_df.columns = ['Transaction']
    return new_df.sort_values(by=['Transaction'])
