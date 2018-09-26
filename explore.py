import dataset


def main():

    data = dataset.Dataset()

    print("Training set")
    print(data.train.describe())

    print("Test set")
    print(data.test.describe())

    print("The most visit times for a customer in train set is: ", 
            explore_utils.find_most_visit(dataset))


if __name__ == '__main__':
    main()
