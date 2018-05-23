import tensorflow as tf


class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate

        self._input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])  # 声明输入变量x, y

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # 训练的时候使用dropout
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

        cell = lstm_cell
        if config.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        # initial states，全部赋值为0状态
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        # define softmax
        softmax_w = tf.get_variable(name="softmax_w", shape=[size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable(name="softmax_b", shape=[vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        # save final state
        self._final_state = state

        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        # 定义损失函数
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        # 获取全部可训练参数
        tvars = tf.trainable_variables()
        # 根据cost计算tvars梯度
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # 定义梯度下降优化器
        optimizer = tf.train.AdamOptimizer(self.lr)
        # 梯度用于可训练参数上
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

    # 装饰器将返回变量设置为只读，防止修改变量引发问题
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op
