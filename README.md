# Stabilized Barzilai-Borwein Step Size
MATLAB MEX implementation of SVRG-SBB algorithms

This code replicates the experiments from the following paper:

- ["Stochastic Non-convex Ordinal Embedding with Stabilized Barzilai-Borwein Step Size". Ke Ma, Jinshan Zeng, Jiechao Xiong, Qianqian Xu, Xiaochun Cao, Wei Liu, Yuan Yao. _AAAI 2018_.]()

Please cite this paper if you use this code in your published research project.

## Requirement
In MATLAB:
``` 
>>mex -v -g -largeArrayDims tste_svrg_bb_epsilon_mex_ifo_time.c
```