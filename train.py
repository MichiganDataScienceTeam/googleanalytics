import argparse
import dataset


def main(args):

    print('Loading dataset...')
    data = dataset.Dataset(debug=args.debug)

    print('Preprocessing...')
    train_df = data.preprocess()

    print('Number of rows:', len(train_df))
    print('Number of columns:', len(train_df.columns))
    print(train_df.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on the Google Analytics Dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    args = parser.parse_args()

    main(args)
