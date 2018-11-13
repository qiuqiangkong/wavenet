### WaveNet pytorch implementation
This code implements speech synthesis with WaveNet using pytorch. This code is partly translated from the tensorflow implementation: https://github.com/ibab/tensorflow-wavenet

## Requirements
pytorch: 0.4.0

## Data preparation
VCTK Corpus includes 44,352 speech sentences uttered by 109 native English speakers. Each utterance last for a few seconds. Download VCTK dataset from https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html. The dataset looks like:

<pre>
dataset_dir
├── wav48
│    ├── p225
│    │     ├── p225_001.wav
│    │     └── ...
│    ├── p226
│    │     ├── p226_001.wav
│    │     └── ...
│    └── ...
├── txt
│    ├── p225
│    │     ├── p225_001.txt
│    │     └── ...
│    ├── p226
│    │     ├── p226_001.txt
│    │     └── ...
│    └── ...
├── speaker-info.txt
└── ...

</pre>

## Run
Modify the paths in runme.sh

Run commands in runme.sh line by line. 

## Results

## FAQ
This code runs wtih a single GPU card with 12 GB memory. If you are running out of GPU memory, then modify Dataset to shorten the audio clip. 

## References
[1] Van Den Oord, Aäron, et al. "WaveNet: A generative model for raw audio." SSW. 2016.
[2] Paine, Tom Le, et al. "Fast wavenet generation algorithm." arXiv preprint arXiv:1611.09482 (2016).
