"""Run benchmark on simple convnet using MNIST """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import flags
import tensorflow as tf

from official.benchmark import keras_benchmark
from official.benchmark import benchmark_wrappers
from official.vision.image_classification import mnist_main

MIN_TOP_1_ACCURACY = 0.929
MAX_TOP_1_ACCURACY = 0.938

FLAGS = flags.FLAGS
MNIST_DATA_DIR_NAME = 'mnist'


class ConvKerasAccuracy(keras_benchmark.KerasBenchmark):
  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    self.data_dir = os.path.join(root_data_dir, MNIST_DATA_DIR_NAME)
    flags_methods = [mnist_main.define_mnist_flags]

    super(ConvKerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flags_methods)

  def _setup(self):
    super(ConvKerasAccuracy, self)._setup()

  def benchmark_graph_1_gpu(self):
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 15
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = mnist_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(ConvKerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)
