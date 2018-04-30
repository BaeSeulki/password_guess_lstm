# coding:utf-8
import datetime
import pickle
import time

import helper
import numpy as np
import tensorflow as tf

from src import Config, Model

from Utils import fn_timer

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
data_dir = config.data_dir
data_size, _vocab_size, char_to_idx, idx_to_char, lines, first_prob = \
    helper.load_code_set(
        max_length=config.MAX_LEN,
        data_dir=data_dir
    )
print('data has %d characters, %d unique.' % (data_size, _vocab_size))
config.vocab_size = _vocab_size

pickle.dump((char_to_idx, idx_to_char, first_prob), open(config.model_path + '.voc', 'wb'), protocol=0)


def inf_train_gen(BATCH_SIZE):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
            yield np.array(
                [[char_to_idx[c] for c in l] for l in lines[i:i + BATCH_SIZE]],
                dtype='int32'
            )


def data_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    batch_len = config.MAX_LEN  # The number of columns of a matrix
    data = raw_data

    epoch_size = (batch_len - 1) // num_steps  # 塞数据的次数

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]  # size： batch_size * num_step
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]  # y与x错一位
        yield (x, y)


def run_epoch(session, m, data, eval_op):
    """Runs the model on the given data."""
    # epoch_size = (config.MAX_LEN-1)// m.num_steps

    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for i in range(config.MAX_LEN - 1):
        x = data[:, i:i + 1]
        y = data[:, i + 1: i + 2]
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     # x和y的shape都是(batch_size, num_steps)
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps
        # print(step)
        # print("%.2f perplexity: %.3f cost-time: %.2f s" %
        #           (step * 1.0 / epoch_size, np.exp(costs / iters),
        #            (time.time() - start_time)))
        # start_time = time.time()

    return np.exp(costs / iters)


def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(is_training=True, config=config)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.global_variables())
        training_star_time = datetime.datetime.now()
        # gen = inf_train_gen(config.batch_size)
        iteration = len(lines) // config.batch_size

        for i in range(iteration):
            temp = []
            for _ in range(config.batch_size):
                line = lines.pop()
                temp.append([char_to_idx[c] for c in line])
            train_data = np.array(temp, dtype=np.int32)
            # train_data = next(gen)
            epoch_time = time.time()
            print("Training Epoch: %d ..." % (i + 1))
            train_perplexity = run_epoch(session, m, train_data, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f cost time:%.3fs"
                  % (i + 1, train_perplexity, time.time() - epoch_time))
            # if i % config.save_freq == config.save_freq-1:
            # print('model saving ...')
        model_saver.save(session, config.model_path +
                         '_' + data_dir[data_dir.rfind('\\') + 1:-4] +
                         '_' + str(config.num_layers) +
                         '_' + str(config.hidden_size))
        # print('Done!')
        print('cost total time:{}'.format(datetime.datetime.now() - training_star_time))


if __name__ == "__main__":
    tf.app.run()

    # print()
