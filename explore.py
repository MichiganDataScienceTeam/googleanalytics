from __future__ import print_function

import dataset
import explore_utils

def main():

    print("Loading the dataset...")
    data = dataset.Dataset()
    
    print("Number of rows in the training set:    ", len(data.train))
    print("Number of columns in the training set: ", len(data.train.columns))

    print("Number of rows in the test set:        ", len(data.test))
    print("Number of columns in the test set:     ", len(data.test.columns))
    
    print("The most visit times for a customer in train set is: ",
          explore_utils.find_most_visit(data))


if __name__ == '__main__':
    main()
