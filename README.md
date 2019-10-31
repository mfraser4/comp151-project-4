# COMP 151 Project 4

Mark Fraser

m_fraser3@u.pacific.edu

# Table of Contents

1. [COMP 151 Project 4](#comp-151-project-4)
2. [Table of Contents](#table-of-contents)
3. [Usage](#usage)
    1. [Args](#args)

# Usage

This project has been split into two programs.  To train a neural network, run
the following command:

```
/path/to/project$ python train.py -c <config>
/path/to/project$ python train.py -h
usage: train.py [-h] [-c CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file (*.ini)
```

The provided configuration file should conform to the following format:

```
[PARAMETERS]
data = <training data file>
learning_rate = <real value (0,+inf)>
epochs = <int value (0,+inf)>
weights = <weights file (optional)>
nn_output = <output for trained neural network (optional)>
```

Note that if no weight output file is provided, the output will automatically be
sent to `output.txt`.

To test a neural network against a set of data, run the following command:

```
path/to/project$ python test.py -w <weights file> -d <data file>
path/to/project$ python test.py -h
usage: test.py [-h] [-w WEIGHTS] [-d DATA]

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS, --weights WEIGHTS
                        Weights file
  -d DATA, --data DATA  Testing data
```

Providing the weights file is optional, but if none is provided, the data will
be trained against a randomly generated neural network.

# Best Testing Output Configuration

Below is the configuration leverage to generate the best neural network:

```
[PARAMETERS]
data = data/training.txt
learning_rate = 0.5
epochs = 5
weights =
nn_output =
```

An example of resulting output (since the weights are randomly generated) is:

```
Aggregate Testing Results:
--------------------------
    Percentage Correct: 0.3733
    Percentage Multiple Firing: 0.16
    Percentage No Firing: 0.3533


Per-Neuron Testing Results:
---------------------------
           Correct   False Pos.  False Neg.
      Red: 0.95      0.01        0.04
     Blue: 0.91      0.0067      0.0833
   Yellow: 0.96      0.03        0.01
    Green: 0.91      0.03        0.06
   Purple: 0.7667    0.19        0.0433
   Orange: 0.9767    0.0         0.0233
    Brown: 0.9233    0.0067      0.07
     Pink: 0.91      0.0         0.09
     Gray: 0.9533    0.0         0.0467
```

The per-neuron statistics are shown to be pretty accurate, while the aggregate
statistics are more muddied.  This is to be expected, however, as colors like
red and pink are likely to have overlapping noise and be non-linearly separable.
This can cause multiple firings, while other noisy data can cause no firing from
any neurons.
