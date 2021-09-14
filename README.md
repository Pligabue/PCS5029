# Main script (index.js)

The main script basically trains 48 different models, varying in the length of their RNN layer, the number of time steps and the number of epochs. To execute it, simply run: 
```sh
node index
```
The script expects the dataset described in `data/README.md`, which should be stored as `data/data.json`.

# Model builder (modelBuilder.js)

The `modelBuilder.js` file contains the functions necessary to train the models. At the end of every training, the model is saved with a (theoretically) unique name, which contains many of its parameters separated by underscores. If there is already a model with a certain set of parameters, the script will simply load it, instead of training a new one. The identification of an already trained model is done by its unique name.

This file also contains a test method, which measures the model's ratio of hits and misses with the same training data. This ratio is recorded on a TXT file inside the model's directory.

# Rank builder (rank.js)

The `rank.js` file simply compiles the hit and miss ratios recorded in the TXT files and builds a CSV file and a JSON file, ranked from best to worst ratio.

# Individual sentence testing (testSentence.js)

The `testSentence.js` file is the other file meant to be executed. It takes command line arguments for the sentence, the expected tags, the size of the RNN layer, the number of time steps and the number of epochs. It uses the script in `modelBuilder.js` to either load or build the model and the inputs (sentence and tags) and outputs (predicted tags and hit ratio) are appended to the `tests.csv` file. Unlike the previous CSV file, this one has to be created by copying the sample `tests.csv.sample` and naming it `tests.csv`. Then, the result of each sentence tested should be append to the end of the file.

An example of how to run this file, with its arguments is"
```sh
node testSentence  -s "What is in the box?" -t WH AUX PREP DT NOUN --rnn 60 --ts 20 -e 300
```
* `-s`: the sentence to be annotated;
* `-t`: the tags, separated by whitespace;
* `--rnn`: the number of units on the RNN layer;
* `--ts`: the number of time steps;
* `-e`: the number of epochs.