# coding:utf-8


class Configs(object):
    # parameters for models
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 15
    num_layers = 3
    num_steps = 30  # number of steps to unroll the RNN
    hidden_size = 32  # size of hidden layer of neurons
    iteration = 30
    save_freq = 5   # the step (counted by the number of iterations) at which the model is saved to hard disk.
    keep_prob = 0.5
    batch_size = 128

    MAX_LEN = 17    # the max len of password (include the end mark)

    # parameters for some file paths
    model_path = '../models/Model'  # the path of model that need to save or load
    data_dir = '../data/train.txt'  # the path of train data
    dictionary_path = '../dictionary/'  # the path of generated data

    # parameters for generation
    save_time = 20  # load save_time saved models
    is_sample = True  # true means using sample, if not using max
    is_beams = True  # whether or not using beam search
    beam_size = 2  # size of beam search
    # len_of_generation = 100  # The number of characters by generated
    start_sentence = '1'  # the seed sentence to generate text

    threshold = 0.0001    # the min prob of generate the password
