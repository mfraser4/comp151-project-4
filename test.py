"""
Mark Fraser
m_fraser3@u.pacific.edu
COMP 151:  Project 4

A color-classification algorithm provided RGB values (0-255) and classification
"""


from argparse import (
    ArgumentParser
)

from parseconfig import (
    parse_config
)

from neuralnetwork import (
    ColorClassifier
)

from sys import (
    exit
)


__author__ = "Mark Fraser"


# main function
#   config - file path of (*.ini) file
def main(args):
    weights_file = args.weights
    data_file = args.data

    # initialize neural network
    nn = ColorClassifier()
    if weights_file is not None and weights_file != "":
        nn.load_weights(weights_file)

    # train neural network on provided data
    if data_file is not None and data_file != "":
        nn.test(data_file)
    else:
        exit('data file must be provided')

    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-w", "--weights", help="Weights file")
    parser.add_argument("-d", "--data", help="Testing data")

    args = parser.parse_args()
    main(args)