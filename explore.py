import dataset


def main():

    data = dataset.Dataset()

    print("Training set")
    print(data.train.describe())

    print("Test set")
    print(data.test.describe())


if __name__ == '__main__':
    main()
