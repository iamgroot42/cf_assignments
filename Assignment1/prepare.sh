#!/bin/bash

wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
rm ml-100k.zip
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d ml-1mn
rm ml-1m.zip
mkdir ml-1m
python k_to_m.py ml-1mn/ ml-1m/
 
