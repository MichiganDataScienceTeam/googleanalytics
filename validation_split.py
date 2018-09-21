from sklearn.model_selection import train_test_split

# The simplist way to split, just use the train_test_split function from library.
# Input: a Dataframe data, the ratio of test set you want, the random state you want.
# Return: two Dataframe.
# This is obvious not a good one, but I'll just write it here first.
def easist_split(data, test_ratio, random_state):
    train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=random_state)
    return train_set, test_set

# test the function
import dataset
data = dataset.Dataset().train
train_set, test_set = easist_split(data, 0.2, 50)

print("Size of train set:", train_set.shape[0])
print("Size of test set:", test_set.shape[0])

# The size of train_set is 722922, test_set is 180731, the ratio is good.