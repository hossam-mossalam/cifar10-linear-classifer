import numpy as np
import tensorflow as tf

from utils import *

class SimpleLinearClassifier:

  def __init__(self, lr = 5e-7, reg = 1e4):
    self.lr = lr
    self.reg = reg
    self._create_graph()

  def _create_placeholders(self):
    with tf.variable_scope('placeholders'):
      self.X = tf.placeholder(tf.float32, shape=[None, 3072], name = 'X')
      self.y = tf.placeholder(tf.int64, shape=[None], name = 'y')

  def _create_variables(self):
    with tf.variable_scope('weights'):
      self.W = tf.Variable(tf.truncated_normal([3072, 10], stddev=1) * 0.0001,
          name = 'W')
      self.b = tf.Variable(tf.zeros([10]), name = 'b')

  def _create_loss(self):
    with tf.name_scope('loss'):
      self.logits = tf.add(tf.matmul(self.X, self.W, name = 'mul'),
                      self.b, name = 'add')
      self.loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels = self.y,
                                                    logits = self.logits),
          name = 'loss')

  def _create_regularization(self):
    with tf.name_scope('regularized-loss'):
      self.loss = self.loss + 0.5 * self.reg * tf.reduce_sum(self.W ** 2)


  def _create_optimizer(self):
    with tf.name_scope('optimizer'):
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

  def _calculate_accuracy(self):
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(self.y,
                                    tf.argmax(self.logits,1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def _create_summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)
      tf.summary.scalar('accuracy', self.accuracy)

      tf.summary.histogram("weights", self.W)
      tf.summary.histogram("biases", self.b)

      self.image_input = tf.reshape(self.X, [-1, 32, 32, 3])
      # self.image_input = tf.reshape(self.X, [-1, 28, 28, 1])
      tf.summary.image('input', self.image_input, 10)

      self.weight_filter = tf.reshape(self.W, [10, 32, 32, 3])
      # self.weight_filter = tf.reshape(self.W, [10, 28, 28, 1])
      filter_summary = tf.summary.image(name = 'filters',
                                        tensor = self.weight_filter,
                                        max_outputs = 10)

      self.summary_op = tf.summary.merge_all()

  def _create_graph(self):
    self._create_placeholders()
    self._create_variables()
    self._create_loss()
    self._create_regularization()
    self._create_optimizer()
    self._calculate_accuracy()
    self._create_summaries()

def train(model, runs = 50000, bs = 32):
  with tf.Session() as sess:

    Xtr, Ytr, Xte, Yte = load_CIFAR10('../data')
    data = prepare_splits(Xtr, Ytr, Xte, Yte)
    X_train, y_train, X_val, y_val, X_dev, y_dev, X_test, y_test = data

    runs = X_train.shape[0] * 5

    X_train = X_train.reshape((-1, 3072))
    X_val = X_val.reshape((-1, 3072))
    X_dev = X_dev.reshape((-1, 3072))
    X_test = X_test.reshape((-1, 3072))

    X_test, y_test = get_minibatch(X_test, y_test, bs)

    sess.run(tf.global_variables_initializer())

    init_loss = sess.run(model.loss,
        feed_dict={model.X: X_test , model.y: y_test})
    print('Initial Loss: ', init_loss)

    writer = tf.summary.FileWriter('model/lr' + str(model.lr), sess.graph)
    for r in range(runs):
      X_current, y_current = get_minibatch(X_train, y_train, bs)

      sess.run([model.train_op],
               feed_dict={model.X: X_current, model.y: y_current})

      if r % 100 == 0:
        print('@ run: ', r)
        summary = sess.run(model.summary_op,
            feed_dict={model.X: X_test , model.y: y_test})

        writer.add_summary(summary, global_step=r)

if __name__ == '__main__':
  linear_classifier = SimpleLinearClassifier()
  train(linear_classifier)
