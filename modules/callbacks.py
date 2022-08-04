import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_freq=500, monitor='myaccuracy'):
        self._current_epoch = 0
        self._batches_since_last_saving = 0
        self.save_freq = save_freq
        self.monitor = monitor

    def on_train_begin(self, logs=None):
        print("Starting training")

    def on_train_end(self, logs=None):
        print("Stop training")

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        print("Start testing")

    def on_test_end(self, logs=None):
        print("Stop testing")

    def on_predict_begin(self, logs=None):
        print("Start predicting")

    def on_predict_end(self, logs=None):
        print("Stop predicting")

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if self._batches_since_last_saving >= self.save_freq:
            self._batches_since_last_saving = 0
            train_acc = logs.get(self.monitor)
            train_loss = logs.get('loss')
            val_acc = logs.get('val_'+self.monitor)
            val_loss = logs.get('val_loss')
            print(" ")
            print("checkpoint:")
            print("train loss = {}, train accuracy = {}, test loss = {}, test accuracy = {}"
                  .format(train_loss, train_acc, val_loss, val_acc))
        else:
            self._batches_since_last_saving += 1


