"""
Mark Fraser
m_fraser3@u.pacific.edu
COMP 151:  Project 4
"""

from configparser import (
    ConfigParser
)

from re import (
    split
)

from sys import (
    exit
)


PARAMETERS = "PARAMETERS"


def parse_config(file):
    config_object = ConfigParser()
    config_object.read(file)

    data_file = get_field(config_object, PARAMETERS, "data")

    eta = float(get_field(config_object, PARAMETERS, "learning_rate"))
    assert eta > 0, 'learning_rate value of {} invalid, must be (0,+inf)'.\
                                                                    format(eta)

    epochs = int(get_field(config_object, PARAMETERS, "epochs"))
    assert epochs > 0, 'epochs value of {} invalid, must be (0,+inf)'.\
                                                                format(epochs)

    weights_file = get_field(config_object, PARAMETERS, "weights")

    out_file = get_field(config_object, PARAMETERS, "nn_output")

    return (data_file, eta, epochs, weights_file, out_file)


def get_field(config, section, field):
    return config[section].get(field)
