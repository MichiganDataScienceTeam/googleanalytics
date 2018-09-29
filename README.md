# googleanalytics

## Running the code

1. Clone this repo
```
git clone https://github.com/MichiganDataScienceTeam/googleanalytics.git
```

2. Download the data from [Google Drive](https://drive.google.com/open?id=1gkD5foFI9vMZzIL_jhDSxiE3T4mGgOqQ) and place it in `./data`

3. Unzip the data and make sure they have read permissions
```
cd data
unzip train.csv.zip
unzip test.csv.zip
unzip sample_submission.csv.zip
chmod +r train.csv test.csv sample_submission.csv.zip
cd ..
```

4. Check to make sure the dataset is in the correct place
```
python dataset.py --debug
```

5. Install the required python packages.
```
pip install -r requirements.txt
```

6. Run the exploration code. Note: removing the `--debug` flag will
cause the full dataset to be loaded, which may take a long time on
your machine.
```
python explore.py --debug
```


## Contributing

1. Create an account on Github and [add an SSH key to your account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/)
2. Ask @stroud on slack to join the [MDST Organization](https://github.com/MichiganDataScienceTeam)
3. Assign yourself to an [issue](https://github.com/MichiganDataScienceTeam/googleanalytics/issues)
4. Create a branch and write your code
5. Submit a pull request when you are done!
