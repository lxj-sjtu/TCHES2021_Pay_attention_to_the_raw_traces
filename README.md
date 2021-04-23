# Pay attention to the raw traces: A deep learning architecture for end-to-end profiling attacks

This repository contains examples for reproducing the results presented in "Pay attention to the raw traces: A deep learning architecture for end-to-end profiling attacks". You could download the paper from https://tches.iacr.org/

In this paper, we approach an end-to-end method to profile and attack the raw traces (over 100,000 time samples) under the protection of masking.
The attack results, i.e., the guessing entropies, are even systematically better than the previous networks trained on the length-reduced traces.

## Repository structure
For each dataset, we give both the script and the best network instance we have found.
For dataset ASCAD and AT128, we consider both the synchronized (syn) and desynchronized (desyn) cases.

## Usage
You could directly run the python script run.py. But we recommend running the code in Jupyter Notebook so that the intermediate plots could be recorded after the training phase.

