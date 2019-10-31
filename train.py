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


__author__ = "Mark Fraser"


# main function
#   config - file path of (*.ini) file
def main(config):
    data_file, eta, epochs, weights_file, out_file = parse_config(config)

    # initialize neural network
    nn = ColorClassifier(eta, epochs)
    if weights_file is not None and weights_file != "":
        nn.load_weights(weights_file)

    # train neural network on provided data
    nn.train(data_file)

    # write output to file if provided
    if out_file is None or out_file == "":
        out_file = 'output.txt' # default output file

    nn.write(out_file)

    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (*.ini)")
    args = parser.parse_args()

    main(args.config)