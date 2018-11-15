import argparse
import dataset


def main(args):

    print('Loading dataset...')
    data = dataset.Dataset(debug=args.debug)

    print('Preprocessing...')
    train_dfs, val_dfs = data.preprocess(do_val_split=True)
    train_df, train_labels = train_dfs
    val_df, val_labels = val_dfs

    print('Number of rows (train):', len(train_df))
    print('Number of rows (val):  ', len(val_df))
    print('Number of columns (train):', len(train_df.columns))
    print('Number of columns (val):  ', len(val_df.columns))

    # Model training goes here!


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on the Google Analytics Dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    args = parser.parse_args()

    main(args)
