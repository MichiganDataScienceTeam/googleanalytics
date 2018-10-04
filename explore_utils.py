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


def find_fraction_of_customers_with_non_zero_revenue(dataset):
    """Find the fraction of customers in the google dataset with non-zero revenue.(only in train set)
    
    args:
        dataset (Dataset): the google analytics dataset.
        
    returns:
            The fraction of customers with non-zero revenue.
    """
    
    
    train_df = dataset.train
    train_df['revenue'] = train_df['totals.transactionRevenue'].astype(float)
    
    """
        # This is a counter variable that increases for each customer with non-zero revenue.
        customer_zero_revenue_count = 0
    
        # for x in the number of rows (customers) of train_df
        for x in train_df:
            if (train_df['revenue'] != 0):
                customer_zero_revenue_count = customer_zero_revenue_count + 1
    
        total_num_customers = len(train_df)
    
        final_fraction = customer_zero_revenue_count / total_num_customers
    
        return final_fraction
    """
    
    # The number of customers who have non-zero revenue.
    num_customers_non_zero = np.count_nonzero(~np.isnan(train_df['revenue']))
    
    # The total number of customers.
    total_num_customers = len(train_df)
    
    return(num_customers_non_zero / total_num_customers)
    
    
    


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

def find_most_common_traffic_sources(dataset,num=5):
    """ Find n most common traffic sources

    args:
       dataset (Dataset): the google analytics dataset
       num(optional): Number of most common traffic sources to return (by default 5)
    returns:
       Series of n (5 by default) most common traffic sources.

    """
    return dataset.train['trafficSource.source'].value_counts().head(num)
