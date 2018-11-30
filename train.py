import argparse
import dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import evaluate
import pickle as pkl
np.random.seed(1267)


def main(args):

    # Cache preprocessing
    try:
      print('Trying to load cached data...')
      train_dfs,val_dfs = pkl.load(open('./_cache/preproceesed_data.pkl','rb'))
      print('Loaded cached data from ./_cache/preproceesed_data.pkl')
    except:
      print ('No cached data from ./_cache/preproceesed_data.pkl')
      print('Loading dataset...')
      data = dataset.Dataset(debug=args.debug)
      print('Preprocessing...')
      train_dfs, val_dfs = data.preprocess(do_val_split=True)
      pkl.dump((train_dfs,val_dfs),open('./_cache/preproceesed_data.pkl','wb'))

    train_df, train_labels = train_dfs
    val_df, val_labels = val_dfs

    print('Number of rows (train):', len(train_df))
    print('Number of rows (val):  ', len(val_df))
    print('Number of columns (train):', len(train_df.columns))
    print('Number of columns (val):  ', len(val_df.columns))

    # Model training goes here!

    # Training Dataset
    #train_df=train_df.dropna(axis=1)
    train_df=train_df.replace(np.nan, -1)
    train_df=train_df.sort_values(by='visitorId')
    train_labels = train_labels.set_index('fullVisitorId')
    train_labels=train_labels.sort_values(by='fullVisitorId')
    

    #val_df=val_df.dropna(axis=1)
    val_df=val_df.replace(np.nan,-1)
    val_df=val_df.sort_values(by='visitorId')
    val_labels=val_labels.set_index('fullVisitorId')
    val_labels=val_labels.sort_values(by='fullVisitorId')


    print ('Training Dataset Columns:',train_df.columns.values)
    print ('Validation Dataset Columns:',val_df.columns.values)

    #train_df=train_df[val_df.columns.values]


    #print(train_df.columns.values)

    #print (train_df.values,train_labels.values)

    # rfc = RandomForestClassifier(n_estimators=500)
    # rfc.fit(train_df.values,train_labels.values.ravel())

    model = MLPClassifier(alpha=1)
    model.fit(train_df.values,train_labels.values.ravel())


    val_pred = model.predict(val_df.values)

    rms = evaluate.rmse_log(val_pred,val_labels.values)

    print('RMS:',rms)

    print('shape:', val_pred.shape)

    rms_zeros = evaluate.rmse_log(np.zeros(val_pred.shape),val_labels.values)

    print('RMS_Zeros:',rms_zeros)



    #val_df=val_df.sort_values(by='visitorId')
    

    #print (val_pred)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model on the Google Analytics Dataset.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    args = parser.parse_args()

    main(args)
