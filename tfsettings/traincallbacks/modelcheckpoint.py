import tensorflow as tf


class ModelCheckpointCallBack:
    def __init__(self, period=1, path=object, monitor='val_los', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto'):
        self.checkpoint_path = ''
        self.period = period
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weight_only = save_weights_only
        self.mode = mode

    def set_checkpoint_path(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        return 0

    def build_cb(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                                 monitor=self.monitor,
                                                                 verbose=self.verbose,
                                                                 save_best_only=self.save_best_only,
                                                                 save_weights_only=self.save_weight_only,
                                                                 mode=self.mode,
                                                                 period=self.period)
        return checkpoint_callback
