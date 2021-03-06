# Pay Attention to Raw Traces: A Deep Learning Architecture for End-to-End Profiling Attacks

This repository contains examples for reproducing the results presented in "Pay Attention to Raw Traces: A Deep Learning Architecture for End-to-End Profiling Attacks". You could download the paper from https://tches.iacr.org/

In this paper, we approach an end-to-end method to profile and attack the raw traces (over 100,000 time samples) under the protection of masking.
The attack results, i.e., the guessing entropies, are even systematically better than the previous networks trained on the length-reduced traces.

## Repository structure
For each dataset, we give both the script and the best network instance we have found.
For datasets ASCAD and AT128, we consider both the synchronized (syn) and desynchronized (desyn) cases.

## Usage
You could directly run the python script run.py. But we recommend running the code in Jupyter Notebook so that the intermediate plots could be recorded after the training phase.

## Getting the AT128
You could get this dataset from [Node_Shanghai](http://47.100.85.38:8000/) or [Node_US_west](http://45.79.106.24:8000/).

Or download it by using wget:
```
wget 47.100.85.38:8000/AT128-N.tar.gz
```
```
wget 47.100.85.38:8000/AT128-F.tar.gz
```

The SHA256 hash values are:

AT128-N.tar.gz

`5b18f9b015d49c9134336e3e9fc2ece813f58c724bf6eac92809670d570bb86a`

AT128-F.tar.gz

`5537428d313e821a826afbde1be4aaf6e3e066c71d6c98c0522f4ee5653baed9`



