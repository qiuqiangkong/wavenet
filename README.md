### WaveNet pytorch implementation
This code implements speech synthesis with WaveNet using pytorch. This code is partly translated from the tensorflow implementation: https://github.com/ibab/tensorflow-wavenet

## Requirements
pytorch: 0.4.0

## Run
Run commands in runme.sh line by line. 

## Results

## FAQ
This code runs wtih a single GPU card with 12 GB memory. If you are running out of GPU memory, then modify Dataset to shorten the audio clip. 

## References
[1] Van Den Oord, Aäron, et al. "WaveNet: A generative model for raw audio." SSW. 2016.
[2] Paine, Tom Le, et al. "Fast wavenet generation algorithm." arXiv preprint arXiv:1611.09482 (2016).
