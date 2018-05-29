import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Animals Clasification")
    parser.add_argument('-clf', "--classifier", dest="clf", type=int, default=1, help='Classifier to train. Default a Convolutional Network.')
    parser.add_argument('-pmode', "--pre_train_mode", dest="pmode", type=str, default='no', help='Pre-train algorithm to use. Default, there is no pre-trainnig')
    parser.add_argument('-rp', "--partitioned", dest="rp", type=int, default=1, help='Number of partitioned networks to stack. Default, 1.')
    parser.add_argument('-d', "--dataset", dest="dataset", type=str, default='avisoft', help='Wich dataset will be used. Deafaul, avisoft')
    parser.add_argument('-f', "--features", dest="features", type=str, default='meanMFCC', help='Features to extract. Default, meanMFCC')
    parser.add_argument('-b', "--build", dest="build", type=int, default=1, help='Weather to load the features or build them. Default, the features will be build')
    parser.add_argument('-it', "--iterations", dest="iterations", type=int, default=1, help='How many times will be trainned the classifier (for statistical purpose). Default, 1')

    return parser.parse_args()


ARGS = parse_args()
