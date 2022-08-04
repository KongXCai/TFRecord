import tensorflow as tf




# define my own accuracy
class MySparseAccuracy(tf.keras.metrics.Metric):
    def __init__(self, batch_size=128, name='myaccuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros', dtype='float32')
        self.total = self.add_weight(name='total', initializer='zeros', dtype='float32')
        self.batch_size = batch_size

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch = tf.cast(self.batch_size, 'float32')
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = (tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        self.total.assign_add(batch)
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

    def reset_state(self):
        self.count.assign(0.1)
        self.total.assign(0.1)


