# coding:utf-8


class Configs(object):
    # parameters for models
    init_scale = 0.04  # 均匀分布初始化的上下界值
    learning_rate = 0.001
    max_grad_norm = 10 # 梯度规范化.最大范数
    num_layers = 1  # 定义多层lstm
    num_steps = 17  # number of steps to unroll the RNN 每次训练多少个字符
    hidden_size = 100  # size of hidden layer of neurons 输出向量ht的维度
    iteration = 30
    save_freq = 5
    keep_prob = 0.5   # rate of dropout while train
    batch_size = 128  # 批大小

    MAX_LEN = 18    # the max len of password (include the end mark)

    # parameters for some file paths
    model_path = '../models/Model'  # the path of model that need to save or load
    data_dir = '../data/17173.txt'  # the path of train data
    dictionary_path = '../dictionary/'  # the path of generated data

    # parameters for generation
    save_time = 30  # load save_time saved models
    is_sample = True  # true means using sample, if not using max
    is_beams = False  # whether or not using beam search
    beam_size = 2  # size of beam search
    len_of_generation = 100  # The number of characters by generated
    start_sentence = '123'  # the seed sentence to generate text

    threshold = 0.00000001    # the min prob of generate the password
