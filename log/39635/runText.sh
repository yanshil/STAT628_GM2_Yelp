#!/bin/bash

# untar your Python installation
tar -xzf anaconda3.tar.gz
# make sure the script will use your Python installation,
# and the working directory as it's home location
export PATH=$(pwd)/anaconda3/bin:$PATH
mkdir home
export HOME=$(pwd)/home

# pip install packages
# pip install googletrans

# run your script
# python3 sklearn_RandomForest.pyi
python3 get_text.py
