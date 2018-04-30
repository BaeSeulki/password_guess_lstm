# coding:utf-8
import datetime
import os
import pickle
from queue import PriorityQueue, Queue
import gc

import numpy as np
import tensorflow as tf

from src import Config, Model
from Utils import fn_timer

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
threshold = config.threshold
data_dir = config.data_dir
char_to_idx, idx_to_char, first_prob = pickle.load(open(config.model_path + '.voc', 'rb'))

config.vocab_size = len(char_to_idx)
is_sample = config.is_sample
is_beams = config.is_beams
beam_size = config.beam_size
len_of_generation = config.len_of_generation
start_sentence = config.start_sentence


def run_epoch(session, m, data, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                                  {m.input_data: x,
                                   m.initial_state: state})
    return prob, _state

@fn_timer
def get_pro(session, mtest, current_prefix):
    print(current_prefix)
    _state = mtest.initial_state.eval()
    for c in current_prefix:
        test_data = np.array([char_to_idx[c]], dtype=np.int32)
        prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
    del _state
    return prob[0]


def main(_):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(config=config_tf, graph=g) as session:
            config.batch_size = 1
            config.num_steps = 1
            # 以上两个参数决定了传入模型数值的size

            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                mtest = Model.Model(is_training=False, config=config)

            # tf.global_variables_initializer().run()

            start=datetime.datetime.now()
            model_saver = tf.train.Saver()
            print('model loading ...')
            model_saver.restore(session, config.model_path +
                                '-' + data_dir[data_dir.rfind('/') + 1:-4] +
                                '-' + str(config.num_layers) +
                                '-' + str(config.hidden_size))
            print('Done!')
            print(datetime.datetime.now()-start)

            # initial prefix list and LUT
            sorted_first_prob = sorted(first_prob.items(), key=lambda x: x[1], reverse=True)
            prefixes = Queue()
            # prefixes = [] #入队按照概率大到小
            LUT = {}  # key-value( prefix-prob )
            for (key, value) in sorted_first_prob:
                prefixes.put(key)
                LUT[key] = value
            # initial prefix list and LUT


            # file operation
            f_id = 1
            generate_path = config.dictionary_path + data_dir[data_dir.rfind('/') + 1:-4] + '-' + str(config.num_layers) + \
                            '-' + str(config.hidden_size) + '/'

            if not os.path.exists(generate_path):
                os.mkdir(generate_path)
            write_star = datetime.datetime.now()
            f = open(generate_path + str(f_id) + '.txt', 'w', encoding='utf-8')
            # file operation


            # initial variable
            code_num = 0
            end_flag = False
            # initial variable

            i = 1
            while not prefixes.empty():
                # current item
                current_prefix = prefixes.get()
                current_prob = LUT[current_prefix]
                LUT.pop(current_prefix)
                # current item

                # get current prefix's next char probability distribution
                # _state = mtest.initial_state.eval()
                # for c in current_prefix:
                #     test_data = np.array([char_to_idx[c]], dtype=np.int32)
                #     prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
                # probability = prob[0]
                probability = get_pro(session, mtest, current_prefix)
                # get current prefix's next char probability distribution

                # sorting the probability distribution
                next_que = PriorityQueue()
                for i in range(len(probability)):
                    next_char = idx_to_char[i]
                    next_prob = probability[i]
                    next_que.put(next_entry(next_char, next_prob))
                # sorting the probability distribution

                # pop the next char
                for i in range(len(probability) // 6):
                    get_next = next_que.get()
                    next_char = get_next.char
                    next_prob = get_next.prob
                    if next_char == "⊥":  # meeting end symbol
                        if len(current_prefix) >= 4:
                            print('第{}条口令为{}，概率为{}'.format
                                  (current_prefix, current_prob, code_num + 1))
                        f.write(current_prefix + '\n')
                        code_num += 1
                        if code_num == 5000000:
                            end_flag = True
                            break
                        if code_num % 1000000 == 0:
                            print('{}.txt耗时：{}'.format(f_id, datetime.datetime.now() - write_star))
                            write_star = datetime.datetime.now()
                            f_id += 1
                            f = open(config.dictionary_path + str(f_id) + '.txt', 'w',
                                     encoding='utf-8')
                    else:
                        new_prefix = current_prefix + next_char
                        if len(new_prefix) == 17:
                            print('第{}条口令为{}，概率为{}'.format
                                  (current_prefix, current_prob, code_num + 1))
                            f.write(current_prefix + '\n')
                            code_num += 1
                            if code_num == 5000000:
                                end_flag = True
                                break
                            if code_num % 1000000 == 0:
                                print('{}.txt耗时：{}'.format(f_id, datetime.datetime.now() - write_star))
                                write_star = datetime.datetime.now()
                                f_id += 1
                                f = open(config.dictionary_path + str(f_id) + '.txt', 'w',
                                         encoding='utf-8')
                        else:
                            new_prob = current_prob * next_prob
                            if new_prob > threshold:
                                LUT[new_prefix] = new_prob
                                prefixes.put(new_prefix)
                # pop the next char

                if end_flag:
                    break

            print('completed')


class next_entry(object):
    def __init__(self, char, prob):
        self.char = char
        self.prob = prob

    # 下面两个方法重写一个就可以了
    def __lt__(self, other):  # 倒序
        return self.prob > other.prob


class prefix_entry(object):
    def __init__(self, prefix, prob):
        self.prefix = prefix
        self.prob = prob

    # 下面两个方法重写一个就可以了
    def __lt__(self, other):  # 倒序
        return self.prob > other.prob


if __name__ == "__main__":
    tf.app.run()
    # print()
