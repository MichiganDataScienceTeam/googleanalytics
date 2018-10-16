from __future__ import division

import pandas as pd
import numpy as np

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


def find_fraction_of_transactions_with_non_zero_revenue(dataset):
    """Find the fraction of transactions in the google dataset with non-zero revenue.(only in train set)
    
    args:
        dataset (Dataset): the google analytics dataset.
        
    returns:
            The fraction of transactions with non-zero revenue.
    """
    
    train_df = dataset.train.copy()
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float)
    
    # The number of transactions that have non-zero revenue.
    num_transactions_non_zero = np.count_nonzero(~np.isnan(train_df['revenue']))
    
    # The total number of transactions.
    total_num_transactions = len(train_df)
    
    return(num_transactions_non_zero / total_num_transactions)


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
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].fillna(0).astype('int64')
    counts = df.groupby('fullVisitorId').first().groupby('channelGrouping').count()
    means = df.groupby('fullVisitorId').first().groupby('channelGrouping')['totals.transactionRevenue'].mean() / 10000
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

def find_return_visit_stats(dataset):
    """Find the statistics of total transactions for returning visitors

    args:
        dataset (Dataset): the google analytics dataset.

    returns:    
        Dataframe of transaction statistics for first time visitors versus return visitors
    """

    train_df = dataset.train.copy()
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float).fillna(0) / 10000
    group_df = (train_df[['fullVisitorId', 'visitNumber', 'revenue']]
          .groupby('fullVisitorId', as_index=False)
          .agg({'visitNumber': 'max', 'revenue': 'sum'}))
    first_stats = (group_df[group_df['visitNumber'] == 1]['revenue']
                  .describe(percentiles=[.95, .99, .999, .9999])
                  .rename('First Time Visitor'))
    return_stats = (group_df[group_df['visitNumber'] != 1]['revenue']
                    .describe(percentiles=[.95, .99, .999, .9999])
                    .rename('Return Visitor'))
    return pd.concat([first_stats, return_stats], axis=1)


def find_revenue_summary_statistics_for_devices(dataset, includeZeroes):
    """Finds summary statistics for revenue generated by each device.

    args:
        dataset (Dataset): the google analytics dataset
        includeZeroes: boolean value, set to True to include sessions with zero
        revenue, set to False to drop sessions with zero revenue

    returns:
        A DataFrame containing first quartile, median, mean, third quartile,
        and standard deviation of revenue generated for each device type

    """

    train_df = dataset.train.copy()
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float).fillna(0) / 10000.0

    if not includeZeroes:
        train_df = train_df[train_df['revenue'] > 0.0]

    train_df['deviceCategory'] = train_df['device.deviceCategory']
    q1 = train_df.groupby('deviceCategory')['revenue'].quantile(0.25)
    median = train_df.groupby('deviceCategory')['revenue'].median()
    mean = train_df.groupby('deviceCategory')['revenue'].mean()
    q3 = train_df.groupby('deviceCategory')['revenue'].quantile(0.75)
    sd = train_df.groupby('deviceCategory')['revenue'].std()

    sum_stat = pd.DataFrame({'q1': q1,
                            'median': median,
                            'mean': mean,
                            'q3': q3,
                            'sd': sd})
    sum_stat.reset_index()

    return sum_stat


def find_percent_sessionIds_using_certain_device(dataset):
    """Finds percent of revenue generating sessionIds that used a particular
    device.

    args:
        dataset (Dataset): the google analytics dataset

    returns:
        A DataFrame containing the percent of all revenue generating sessions that
        used a particular device

    """
    train_df = dataset.train.copy()

    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float).fillna(0) / 10000.0
    train_df = train_df[train_df['revenue'] > 0]
    train_df['deviceCategory'] = train_df['device.deviceCategory']
    counts = train_df.groupby('deviceCategory')['deviceCategory'].count()
    counts_df = pd.DataFrame(counts)
    counts_df = counts_df = counts_df.rename(index = str, columns = {'deviceCategory':'count'})
    counts_df['percent'] = (counts_df['count'] / counts_df['count'].sum()) * 100.0
    percent_df = counts_df.drop('count', axis = 1).reset_index()

    return percent_df


def find_percent_of_total_revenue_by_device(dataset):

    """Finds percent of total revenue generated that is attributable to
    particular devices.

    args:
        dataset (Dataset): the google analytics dataset

    returns:
        A DataFrame containing the percent of total revenue generated by sessions
        that used a particular device

    """
    train_df = dataset.train.copy()

    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float).fillna(0) / 10000.0
    train_df['deviceCategory'] = train_df['device.deviceCategory']
    revenue_df = pd.DataFrame(train_df.groupby('deviceCategory')['revenue'].sum())
    revenue_df['percent'] = revenue_df['revenue'] / revenue_df['revenue'].sum()
    percent_df = revenue_df.drop('revenue', axis = 1).reset_index()
    percent_df['percent'] = percent_df['percent'] * 100

    return percent_df

def find_percent_device_uses_generating_revenue(dataset):

    """Finds percent of sessions using a particular devices
    that generated revenue.

    args:
        dataset (Dataset): the google analytics dataset

    returns:
        A series containing the percent of sessions that used a particular
        device that actually generated revenue

    """
    train_df = dataset.train.copy()

    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float).fillna(0)
    train_df['deviceCategory'] = train_df['device.deviceCategory']
    train_df['generatingRevenue'] = train_df['revenue'] > 0.0
    train_df['notGeneratingRevenue'] = train_df['revenue'] == 0.0

    genRev = train_df.groupby('deviceCategory')['generatingRevenue'].sum()
    genRev_df = pd.DataFrame(genRev).reset_index()

    nonGenRev = train_df.groupby('deviceCategory')['notGeneratingRevenue'].sum()
    nonGenRev_df = pd.DataFrame(nonGenRev).reset_index()

    genRev_df['percent'] = genRev_df['generatingRevenue'] / (nonGenRev_df['notGeneratingRevenue'] + genRev_df['generatingRevenue'])
    percent_df = genRev_df.drop('generatingRevenue', axis = 1)
    percent_df['percent'] = percent_df['percent'] * 100

    return percent_df

