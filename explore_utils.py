def find_most_visit(dataset):
    """Find what is the most visited times of single customer.(only in train set)

    args:
        dataset (Dataset): the google analytics dataset.

    returns:
	    The most visited times in trainset.
    """

    train_df = dataset.train.copy()

    train_gdf = train_df.groupby("fullVisitorId")['visitNumber'].sum().reset_index()
    max_visit = train_gdf['visitNumber'].max()
	
    return max_visit
