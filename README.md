# income-prediction
predicting the occupation and income of Twitter users using graph embeddings. This is code acompanying the paper 'Predicting Twitter User Socioeconomic Attributes with Network and Language Information' appearing at ACM Hypertext 2018.

These instructions describe how to build graph embeddings for a sample of the Twitter network that have been labelled with incomes and occupations. 

## Getting Started

To run the model clone the repo 

cd to the project's root folder

python src/python/generate_embeddings.py resources/X_thresh10.p <out_path>

### Prerequisites

The code uses the numpy, pandas and scikit-learn python packages. We recommend installing these through Anaconda

### Data

For privacy reasons we can't include the raw Twitter data. Instead we include a pickled network file in resources/X_thresh10.p

This file is a pickled scipy sparse matrix containing the the ego-networks of all users that have income / occupation labels as described in the paper, but thresholded to only include accounts with at least 10 connections.

To read the data:

import pandas

x = pd.read_pickle('X_thresh10.p')

To increase the general utility of the code, we also include the income lables as income_y.p in the resources folder, which is a pandas pickle file of a pandas dataframe.

## Authors

**Ben Chamberlain**
**Nikolaos Aletras**

## Citation

If this code is useful to you, please cite:

Nikolaos Aletras and Benjamin Paul Chamberlain. "Predicting Twitter User Socioeconomic Attributes with Network and Language Information", ACM HT18 2018.
