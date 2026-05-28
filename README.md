# CNN Exoplanet Classifier Using Transit Light Curves

This code is utilized for the research of exoplanet transit light curves. 

## Overview

Currently, there are neural networks utilized for taking time series flux data and classifying them as potential
exoplanets or common false positives. These require heavy processing of data. This project attempts to show we can use 
convolutional neural networks for these classifications with minimal preprocessing.

Data from strictly the Kepler mission is used for this work as it has large and complete data sets useful for learning 
off of. Thus, this trained algorithm is best suited for Kepler data but may be enhanced to work on other missions such 
as TESS or PLATO.  

A simple convolutional model is used here with two convolutional layers and dropout used to restrict overfitting.
## Project Structure

```text
configs/            YAML configuration files for data processing and model settings
data/raw/           Raw input catalogues
data/processed/     Processed light curve datasets
data/real_model/    Train/validation/test NumPy datasets
models/             Trained model files
notebooks/          Exploratory research notebooks
scripts/            Reproducible Python pipeline scripts
results/            Metrics and evaluation outputs
```