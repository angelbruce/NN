from mnist import *
from cifar import *


"""  basic config path for training or testing """
class config(object):
    """  basic config path for training or testing """

    use_cudnn_on_gpu = True

    model_checkpoint_name = "model.ckpt"
    model_checkpoint_base = "/home/abl/workspace/python/model/"


    mnist_train_traits = "/home/abl/workspace/python/data/mnist/train-images-idx3-ubyte"
    mnist_train_labels = "/home/abl/workspace/python/data/mnist/train-labels-idx1-ubyte"
    mnist_test_traits = "/home/abl/workspace/python/data/mnist/t10k-images-idx3-ubyte"
    mnist_test_labels = "/home/abl/workspace/python/data/mnist/t10k-labels-idx1-ubyte"

    cifar_data_path = "/home/abl/workspace/python/data/cifar/"
    

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

    @staticmethod
    def cifar_train_reader(name):
        """ get the cifar data reader for training """
        return cifar(config.cifar_data_path,config.check_points(name))

    @staticmethod
    def cifar_test_reader(name):
        """ get the cifar data reader for testing """
        return cifar(config.cifar_data_path,config.check_points(name),is_test=True)
