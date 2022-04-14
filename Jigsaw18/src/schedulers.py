import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


@tf.keras.utils.register_keras_serializable()
class GradualWarmupScheduler(LearningRateSchedule):
    def __init__(self,
                 initial_learning_rate,
                 warmup_steps,
                 end_learning_rate,
                 after_scheduler,
                 name=None):
        super(GradualWarmupScheduler, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.after_scheduler = after_scheduler
        self.name = name

    def warmup_decay(self, step):
        initial_learning_rate = tf.convert_to_tensor(
            self.initial_learning_rate)
        dtype = initial_learning_rate.dtype
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        end_learning_rate = tf.cast(self.end_learning_rate, dtype)

        global_steps_recomp = tf.cast(step, dtype)

        p = global_steps_recomp / warmup_steps
        return tf.add(
            tf.multiply(end_learning_rate - initial_learning_rate, p),
            initial_learning_rate)

    def __call__(self, step):
        with tf.name_scope(self.name or "GradualWarmupScheduler") as _:
            warmup_steps = tf.convert_to_tensor(
                self.warmup_steps, name="warmup_steps")
            dtype = warmup_steps.dtype

            global_steps_recomp = tf.cast(step, dtype)

            learning_rate = tf.cond(
                global_steps_recomp >= warmup_steps,
                lambda: self.after_scheduler(global_steps_recomp - warmup_steps),
                lambda: self.warmup_decay(global_steps_recomp)
            )
            return learning_rate

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'end_learning_rate': self.end_learning_rate,
            'after_scheduler': self.after_scheduler,
            'name': self.name
        }
