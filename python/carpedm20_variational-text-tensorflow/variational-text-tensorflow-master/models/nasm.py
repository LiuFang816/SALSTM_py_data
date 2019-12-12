import time
import tensorflow as tf

from base import Model

class NASM(Model):
  """Neural Answer Selection Model"""

  def __init__(self, sess, reader, dataset="ptb",
               batch_size=20, num_steps=3, embed_dim=500,
               h_dim=50, learning_rate=0.01, epoch=50,
               checkpoint_dir="checkpoint"):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: The name of dataset to use.
      h_dim: The dimension of document representations (h). [50, 200]
    """
    self.sess = sess
    self.reader = reader

    self.h_dim = h_dim
    self.embed_dim = embed_dim

    self.epoch = epoch
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir

    self.dataset="ptb"
    self._attrs=["batch_size", "num_steps", "embed_dim", "h_dim", "learning_rate"]

    raise Exception(" [!] Working in progress")
    self.build_model()

  def build_model(self):
    self.q = tf.placeholder(tf.float32, [self.reader.vocab_size], name="question")
    self.a = tf.placeholder(tf.float32, [self.reader.vocab_size], name="answer")

    self.build_encoder()
    self.build_decoder()

    # Kullback Leibler divergence
    self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))

    # Log likelihood
    self.g_loss = tf.reduce_sum(tf.log(self.p_x_i))

    self.loss = tf.reduce_mean(self.e_loss + self.g_loss)
    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(-self.loss)

    _ = tf.scalar_summary("encoder loss", self.e_loss)
    _ = tf.scalar_summary("decoder loss", self.g_loss)
    _ = tf.scalar_summary("loss", self.loss)

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    with tf.variable_scope("encoder"):
      q_cell = tf.nn.rnn_cell.LSTMCell(self.embed_dim, self.vocab_size)
      a_cell = tf.nn.rnn_cell.LSTMCell(self.embed_dim, self.vocab_size)

      l1 = tf.nn.relu(tf.nn.rnn_cell.linear(tf.expand_dims(self.x, 0), self.embed_dim, bias=True, scope="l1"))
      l2 = tf.nn.relu(tf.nn.rnn_cell.linear(l1, self.embed_dim, bias=True, scope="l2"))

      self.mu = tf.nn.rnn_cell.linear(l2, self.h_dim, bias=True, scope="mu")
      self.log_sigma_sq = tf.nn.rnn_cell.linear(l2, self.h_dim, bias=True, scope="log_sigma_sq")

      eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
      sigma = tf.sqrt(tf.exp(self.log_sigma_sq))

      _ = tf.histogram_summary("mu", self.mu)
      _ = tf.histogram_summary("sigma", sigma)

      self.h = self.mu + sigma * eps

  def build_decoder(self):
    """Inference Network. p(X|h)"""
    with tf.variable_scope("decoder"):
      R = tf.get_variable("R", [self.reader.vocab_size, self.h_dim])
      b = tf.get_variable("b", [self.reader.vocab_size])

      x_i = tf.diag([1.]*self.reader.vocab_size)

      e = -tf.matmul(tf.matmul(self.h, R, transpose_b=True), x_i) + b
      self.p_x_i = tf.squeeze(tf.nn.softmax(e))

  def train(self, config):
    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    for epoch in range(self.epoch):
      epoch_loss = 0.

      for idx, x in enumerate(self.reader.next_batch()):
        _, loss, e_loss, g_loss, summary_str = self.sess.run(
            [self.optim, self.loss, self.e_loss, self.g_loss, merged_sum], feed_dict={self.x: x})

        epoch_loss += loss
        if idx % 10 == 0:
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, e_loss: %.8f, g_loss: %.8f" \
              % (epoch, idx, self.reader.batch_cnt, time.time() - start_time, loss, e_loss, g_loss))

        if idx % 2 == 0:
          writer.add_summary(summary_str, step)

        if idx != 0 and idx % 1000 == 0:
          self.save(self.checkpoint_dir, step)

  def sample(self, sample_size=10):
    """Sample the documents."""
    sample_h = np.random.uniform(-1, 1, size=(self.sample_size , self.h_dim))

    _, = self.sess.run([self.p_x_i], feed_dict={self.x: x})
