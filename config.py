from mnist import *


"""  basic config path for training or testing """
class config(object):
    """  basic config path for training or testing """

    use_cudnn_on_gpu = True

    model_checkpoint_name = "model.ckpt"
    model_checkpoint_base = "/home/abl-0810/ai/model/"


    mnist_train_traits = "/home/abl-0810/ai/data/train-images.idx3-ubyte"
    mnist_train_labels = "/home/abl-0810/ai/data/train-labels.idx1-ubyte"

    mnist_test_traits = "/home/abl-0810/ai/data/t10k-images.idx3-ubyte"
    mnist_test_labels = "/home/abl-0810/ai/data/t10k-labels.idx1-ubyte"

    @staticmethod
    def check_points(name):
        """save the check-point-store-path form train model or get the  check-point-store-path for test model"""
        return config.model_checkpoint_base + name +"/" + config.model_checkpoint_name

    @staticmethod
    def mnist_train_reader(name):
        """ get the minist data reader for training """
        return mnist(config.mnist_train_traits, config.mnist_train_labels,
                     config.check_points(name))

    @staticmethod
    def mnist_test_reader(name):
        """ get the minist data reader for testing """
        return mnist(config.mnist_test_traits, config.mnist_test_labels,
                     config.check_points(name))
