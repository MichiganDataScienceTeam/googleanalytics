import numpy as np

def revenue_per_trafficsource(dataset):
    """Find the average transaction revenue by traffic Source

    args :
         dataset (Dataset): the google analytics Dataset

    returns:
         Dataframe of average transaction revenue by traffic Source
    """
    train_df = dataset.train.copy()
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float)
    train_df['source'] = train_df['trafficSource.source']
    train_df = train_df[['revenue','source']]
    train_df = train_df.fillna(0)
    result = train_df.groupby('source')['revenue'].mean()/10000
    return result


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

    train_df = dataset.train
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float)
    revenue_per_customer = train_df.groupby('fullVisitorId')['revenue']
    total_revenue_per_customer = revenue_per_customer.sum().fillna(0) / 10000.0

    values = [np.percentile(total_revenue_per_customer.values, percentile)
              for percentile in percentiles]

    return values
