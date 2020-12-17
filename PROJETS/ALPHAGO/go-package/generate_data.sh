#!/bin/bash

for i in {0..100}
do
    rm data/100_iter.json || 1
    python3 namedGame.py
    python3 train_policy_model.py
done
