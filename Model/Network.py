import tensorflow as tf
import util.tf_utils as tf_utils

class ComposeNetwork:
    def __init__(self, input_shape, num_classes, learning_rate = 1e-3, sess = None, ckpt_path = None, gpu_mode=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.gpu_mode = gpu_mode

        self.sess = sess
        self.__build_net__()

        if ckpt_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_path)

    def __build_net__(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None,]+self.input_shape)
        self.y = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.dropout = tf.placeholder(dtype=tf.float32, shape=())
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=())

        self.Y = tf.one_hot(self.y, depth=self.num_classes, axis=-1)

        x = tf.transpose(self.X,[1,0,2])
        with tf.name_scope('LSTM') and tf_utils.set_device_mode(self.gpu_mode):
        #if True:
            L0, _ = tf_utils.LSTMlayer(x, 256, self.batch_size,0, self.gpu_mode)
            L0 = tf.nn.dropout(L0, keep_prob=1-self.dropout)

            L1, _ = tf_utils.LSTMlayer(L0, 512, self.batch_size, 1, self.gpu_mode)
            L1 = tf.nn.dropout(L1, keep_prob=1-self.dropout)

            L2, _ = tf_utils.LSTMlayer(L1, 256, self.batch_size, 2, self.gpu_mode)
            L2 = tf.nn.dropout(L2, keep_prob=1-self.dropout)
            L2 = L2[-1]
        #print(L2.shape)
        with tf.name_scope('MLP') and tf_utils.set_device_mode(self.gpu_mode):
            MLP1 = tf_utils.Dense(L2, 256, name='M1' ,activation=None)
            MLP1 = tf.nn.dropout(MLP1, keep_prob=1-self.dropout)

            MLP2 = tf_utils.Dense(MLP1, 128, name='M2',activation=None)
            MLP2 = tf.nn.dropout(MLP2, keep_prob=1 - self.dropout)

            self.logit = tf_utils.Dense(MLP2, self.num_classes, 'logit')
            self.out = tf.nn.softmax(self.logit)

            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logit,
                labels=self.Y,
            )
        self.loss = tf.reduce_mean(self.loss)
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


    def train(self, x, y, dropout = 0):
        return self.sess.run(
            [self.optim, self.loss],
            feed_dict={self.X: x, self.y: y, self.batch_size: len(x), self.dropout:dropout}
        )[1]


    def eval(self, x, y):
        out = self.sess.run(
            tf.argmax(self.out, axis = 1),
            feed_dict={self.X: x, self.batch_size: len(x), self.dropout:0.0}
        )
        TF = [out[i]==y[i] for i in range(len(y))]
        summed = sum(TF)
        return summed/len(y)

    def infer(self,x):
        return self.sess.run(
            tf.argmax(self.out, axis = 1),
            feed_dict={self.X: x, self.batch_size: len(x), self.dropout: 0.0}
        )

if __name__ == '__main__':
    mock_x = [[[0],[0],[0]],[[0],[0],[0]],[[0],[0],[0]]]
    mock_y = [1,2,5]
    net = ComposeNetwork(
        input_shape=[3,1],
        num_classes=6,
        sess = tf.Session()
    )

    print(net.train(mock_x,mock_y))
    print(net.eval(mock_x,mock_y))
    print(net.infer([mock_x[0]]))
