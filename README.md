STATUS as of 4-18:

Have a running implementation of parallel NN code, but very slow (even with 1 thread) and possibly incorrect - most of the time is being spent in the backward pass function of the neural network force computation

Currently - seeing that non NN system is getting NaN forces after a couple hundred timesteps
Replaced analytical force computation with compute grad in order to automatically take care of shape/masking issues

TODO: 
- verify that the new parallel/serial  code is correct for poly and non poly systems, and for NN and non NN - generate "ground truths" for each of these by running code from the other branches (SR)

- start making the poster slide (CA)

- figure out why multiple threads is so slow - is it because of the serial appending of the forces ?








