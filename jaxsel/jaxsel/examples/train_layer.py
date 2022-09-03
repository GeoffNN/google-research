# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo of functionality around learned subgraph extraction.

Trains a subgraph extraction layer by strong supervision.
"""

import functools
import os
import warnings

from absl import app
from absl import flags

import jax
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

import flax.training.checkpoints

import numpy as np
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

import tqdm

import shortuuid

from jaxsel._src import agents
from jaxsel._src import image_graph
from jaxsel._src import subgraph_extractors
from jaxsel._src import train_utils
from jaxsel._src import tree_utils
from jaxsel.examples.utils import data as data_utils


FLAGS = flags.FLAGS

# TODO(gnegiar): Use return values from flags.
flags.DEFINE_enum('dataset', 'mnist', ['mnist', 'lra_pathfinder'],
                  'which dataset to use')
flags.DEFINE_enum('pathfinder_difficulty', 'easy', ['easy', 'hard'],
                  'The level of difficulty for the pathfinder dataset.')
flags.DEFINE_integer('pathfinder_resolution', 32,
                     'Resolution for the pathfinder task.')
flags.DEFINE_integer('batch_size', 32, 'Training batch size')
flags.DEFINE_integer('log_freq', 1,
                     'Log batch accuracy and loss every log_freq iterations.')
flags.DEFINE_integer('plot_freq', 100,
                     'Log image plots for a random train examples and first val points for each class every plot_freq iterations.')
flags.DEFINE_integer('n_epochs', 5, 'Number of training epochs.')
flags.DEFINE_float(
    'alpha', 2e-4,
    'Probability of teleporting back to initial node distribution.')
flags.DEFINE_float('rho', 0., 'L1 regularization in sparse PageRank.')
flags.DEFINE_integer(
    'patch_size',
    5,
    'Size of the patch used to compute node features.',
    lower_bound=1)
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'sgd'], 'optimizer to use')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate for the optimizer.')

flags.DEFINE_string(
    'tensorboard_logdir', os.environ.get('AIP_TENSORBOARD_LOG_DIR', None),
    'Path to directory where tensorboard summaries are stored.')
flags.DEFINE_bool(
    'use_node_weights', True,
    'Whether to use the learned node weights in the downstream graph model.')
flags.DEFINE_integer('max_subgraph_size', 100,
                     'Maximum allowed size for the extracted subgraph.')
flags.DEFINE_integer('num_steps_extractor', 25,
                     'Number of ISTA steps for extractor forward.')
flags.DEFINE_bool('debug', False,
                  'If True, we only train over 1 batch before testing.')
flags.DEFINE_float(
    'ridge_backward', 1e-6,
    'L2 regularization for the linear system used to compute'
    'the jvp of the subgraph selection layer.')

flags.DEFINE_integer('num_heads', 4, 'Number of heads for attention mechanism.')
flags.DEFINE_integer('qkv_dim', 128, 'Attention mechanism dimension.')
flags.DEFINE_integer('mlp_dim', 16,
                     'Dense layer dimension in transformer block.')
flags.DEFINE_integer('agent_hidden_dim', 16,
                     'Middle layer dimension for agent model.')
flags.DEFINE_integer('agent_hidden_combination_dim', 32,
                     'Dimension of combination layer for agent model.')
flags.DEFINE_integer('graph_model_hidden_dim', 32,
                     'Dimension of graph model hidden layer.')
flags.DEFINE_integer('feature_embedding_dim', 5,
                     'Embedding dimension for agent model.')
flags.DEFINE_integer('n_encoder_layers', 5,
                     'Number of layers in graph model encoder.')
flags.DEFINE_integer('max_graph_size', 900,
                     'Max graph size for subgraph extractor.')
flags.DEFINE_bool(
    'overfit', False,
    'if True, use only 1st validation batch, to check whether we can overfit to them.'
)
flags.DEFINE_bool(
    'supernode', False, 'if True, adds a supernode to the selected subgraph. '
    'The supernode is connected to all nodes.')
flags.DEFINE_bool('local', False, 'Pass if the run is local')
flags.DEFINE_integer('test_log_freq', 1000, 'Log test statistics every n batches.')
flags.DEFINE_integer('seed', 0, 'Seed for the random number generator.')

metrics = tf.keras.metrics


def training_loop(
    optimizer,
    patch_size,
    batch_size,
    dataset,
    pathfinder_resolution,
    pathfinder_difficulty,
    num_heads,
    qkv_dim,
    mlp_dim,
    graph_model_hidden_dim,
    learning_rate,
    alpha,
    rho,
    max_subgraph_size,
    max_graph_size,
    n_epochs,
    ridge,
    tensorboard_logdir,
    num_steps_extractor,
    agent_hidden_dim,
    n_encoder_layers,
    supernode=False,
    overfit=False,
    debug=False,
    log_freq=1,
    plot_freq=20,
    test_log_freq=1,
    seed=0,
):
  """Image classification training loop.


  Args:
    optimizer: Which optimizer to use. For now, ['adam', 'sgd'].
    patch_size: width of the patch used to make node features
    batch_size: batch size for both train and validation sets.
    dataset: which dataset to use.
    pathfinder_resolution: Resolution for pathfinder task. 32, 64 or 128.
    pathfinder_difficulty: Difficulty of pathfinder task. Must be in ['easy',
      'intermediate', 'hard']
    num_heads: Number of heads in multi head attention.
    qkv_dim: Attention mechanism dimension.
    mlp_dim: Dense layer dim in attention.
    graph_model_hidden_dim: Hidden dim for the message passing graph model.
    learning_rate: learning rate of the optimizers
    alpha: probability of teleporting back to the start node in the subgraph
      selection
    rho: scale of the sparsity penalty for subgraph selection
    max_subgraph_size: maximum allowed size for the extracted subgraph
    max_graph_size: maximum allowed size for the graph
    n_epochs: number of training epochs
    ridge: scale of ridge regularization for the backwards of the subgraph
      selection layer.
    tensorboard_logdir: Directory path for tensorboard logs.
    num_steps_extractor: Number of iterations for the subgraph extractor.
    agent_hidden_dim: Embedding dimension for task, nodes and edges in the agent
      model.
    n_encoder_layers: Depth of the graph encoder model.
    supernode: whether to add a supernode to the extracted subgraph. The
      supernode is connected to all ndoes in the subgraph, and is an attempt to
      get by range issues on the subgraph.
    overfit: option to only consider 2 samples
    debug: If True, only perform 1 batch in train and 1 epoch total. Allows to
      debug both train and test.
    log_freq: Log train statistics every N batches.
    plot_freq: Log image plots for a random train examples and first val points 
      for each class every plot_freq iterations.
    test_log_freq: Compute test statistics every N epochs.
    seed: Seed for the random number generator.
  """

  # TODO(gnegiar): Refactor this in utils.data
  # Load data
  if dataset == 'mnist':
    (train_dataset, test_dataset, image_shape,
     num_classes) = data_utils.load_mnist(batch_size)
    val_dataset = None
    make_graph = data_utils.make_graph_mnist
    bins = jnp.array([0., .3, 1.])

  elif dataset == 'lra_pathfinder':
    (train_dataset, val_dataset, test_dataset, image_shape,
     num_classes) = data_utils.load_pathfinder(batch_size,
                                               pathfinder_resolution,
                                               pathfinder_difficulty, overfit)

    test_dataset = val_dataset  # Drop the test set, validate on val_ds
    test_dataset = test_dataset.cache()
    make_graph = data_utils.make_graph_pathfinder
    # TODO(gnegiar): take bins as argument ?
    bins = jnp.linspace(0., 1., 50)

  else:
    raise ValueError(
        f"dataset must be in ['mnist', 'lra_pathfinder']. Got {dataset}.")

  # TODO(gnegiar): Add gradient clipping
  if optimizer == 'adam':
    optimizer = optax.adam(learning_rate)
  elif optimizer == 'sgd':
    optimizer = optax.sgd(learning_rate)

  # All the graphs generated in the training loop share graph_parameters
  # This initializes graph_parameters,
  # used to initialize the Agent and Graph models.
  graph = image_graph.ImageGraph.create(
      jnp.ones(image_shape, dtype=int),
      lambda _: (14, 14),  # doesn't matter for initializing the model
      patch_size=patch_size,
      # binarized features + out of bounds pixel (+ supernode)
      num_colors=len(bins) + 2 if supernode else len(bins) + 1)
  
  graph_parameters = graph.graph_parameters()

  # Initialize models
  rng = jax.random.PRNGKey(seed)

  # TODO(gnegiar): Define configs outside of training loop, and pass as args.
  agent_config = agents.AgentConfig(graph_parameters, agent_hidden_dim,
                                    agent_hidden_dim)

  extractor_config = subgraph_extractors.ExtractorConfig(
      max_graph_size, max_subgraph_size, rho, alpha, num_steps_extractor, ridge,
      agent_config)


  # Define loss over the full model.
  def forward(params, graphs, start_node_ids, config, rng=None):
    extractor = subgraph_extractors.SparseISTAExtractor(config)
    q_star, node_features, node_ids, dense_submat, q, adjacency_matrix, error = jax.vmap(extractor.apply, (None, 0, 0))(params, start_node_ids, graphs)
    del dense_submat, adjacency_matrix, error, node_features
    image_shape = graphs.image.shape[1:]
    graph_size = graphs.image[0].squeeze().size
    # Get q as an image and normalize
    batch_size = q_star.shape[0]
    q_images = jnp.zeros((batch_size, graph_size))
    q_images = q_images.at[jnp.arange(batch_size), node_ids.T].set(q_star.T)
    q_images = q_images.reshape(batch_size, *image_shape)
    q_images_norm = q_images / q_images.sum(axis=(1, 2), keepdims=True)
    # Normalize image
    images = jnp.clip(graphs.image - 1, a_min=0)  # get rid of the shift due to the -1 node
    images_norm = images / images.sum(axis=(1, 2), keepdims=True)
    # Favor black pixels
    diff_norm = (q_images_norm - images_norm).reshape(-1, graph_size)
    loss_vals = (diff_norm ** 2).sum(-1)
    # loss_vals = -((jnp.abs(q_images_norm * images_norm)).reshape(-1, graph_size) ** (1/3)).sum(-1)
    # Penalize white pixels
    # loss_vals -= jnp.abs(q_images_norm - (images_norm == 0)).reshape(-1, graph_size).sum(-1)
    # TODO(gnegiar): Try L2 with different weights for black/white pixels 

    return loss_vals.mean(0), q

  forward = functools.partial(forward, config=extractor_config)
  forward = jax.jit(forward)

  # differentiate wrt model parameters
  value_grad_loss_fn = jax.value_and_grad(forward, has_aux=True)
  value_grad_loss_fn = jax.jit(value_grad_loss_fn)

  # Set apart first test datapoints of each class for visualizing.
  test_representatives = train_utils.get_first_class_representatives(
      test_dataset, num_classes)
  test_rep_images, _ = zip(*test_representatives)
  init_graph = make_graph(test_representatives[0][0], patch_size, bins)
  test_representatives_graphs = tree_utils.tree_stack([
      make_graph(image, patch_size, bins)
      for image, label in test_representatives
  ])
  rep_labels = np.arange(num_classes)

  rng_params, rng_dropout, rng = jax.random.split(rng, 3)

  model = subgraph_extractors.SparseISTAExtractor(extractor_config)
  model_state = model.init(
      {
          'params': rng_params
      },
      init_graph.sample_start_node_id(),
      init_graph,
  )

  opt_state = optimizer.init(model_state)

  # Initialize tensorboard logger
  if tensorboard_logdir is not None:
    tf_logger_train = tf.summary.create_file_writer(
        os.path.join(tensorboard_logdir, 'train'))

    tf_logger_test = tf.summary.create_file_writer(
        os.path.join(tensorboard_logdir, 'test'))

  # Main training loop

  if debug:
    n_epochs = 1

  step = 0

  # For early stopping
  best_loss = jnp.inf
  best_test_loss = jnp.inf

  print('Start training!')
  # TODO(gnegiar): Add description with loss to tqdm
  for epoch in tqdm.tqdm(range(n_epochs)):
    for batch in tfds.as_numpy(train_dataset):
      data, labels = batch
      # TODO(gnegiar): build the graphs once before hand, in the dataloading
      # Make graphs from the batch of images
      graphs = tree_utils.tree_stack(
          [make_graph(image, patch_size, bins) for image in data])

      rng_it, rng = jax.random.split(rng)

      (loss, q), model_grad = value_grad_loss_fn(
          model_state,
          graphs,
          graphs.sample_start_node_id(),
          rng=rng_it,
      )
      print(loss)

      if np.isnan(loss):
        warnings.warn('Train loss was NaN.', RuntimeWarning)

      elif loss < best_loss:
        best_loss = loss

      # Log the loss
      if tensorboard_logdir is not None:
        if step % log_freq == 0:

          # TODO(gnegiar): Separate out the gradients for both models
          agent_model_grad_norm = tree_utils.global_norm(
              model_grad['params'])

          with tf_logger_train.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar(
                'agent_model_grad_norm', agent_model_grad_norm, step=step)

        if step % plot_freq == 0:
          with tf_logger_train.as_default():
            tf.summary.image(
                'Training data',
                train_utils.plot_to_image(
                    figure=train_utils.plot_subgraph(
                        img=data[0].squeeze(),
                        # Select first element in batch of qs
                        q=jsparse.BCOO((q.data[0], q.indices[0]),
                                       shape=q.shape[1:]),
                        label='N/A',
                        # Transposing usual batch convention due to flax
                        start_node_coords=graphs.start_node_coords[:, 0])),
                step=step)
            tf.summary.histogram('Node weights', q.data[0], step=step)

            # Plot subgraph on first test datapoint of each class
            loss_rep, q_rep = forward(
                model_state, test_representatives_graphs,
                test_representatives_graphs.sample_start_node_id()
                )
            preds_rep = ['N/A'] * 10
            del loss_rep
            tf.summary.image(
                'Class representatives',
                train_utils.plot_to_image(
                    train_utils.plot_subgraph_classes(
                        test_rep_images,
                        q_rep,
                        preds_rep,
                        # transpose because of weird flax behavior
                        test_representatives_graphs.start_node_coords.T,
                        num_classes)),
                step=step)

      pipeline_update, opt_state = optimizer.update(model_grad, opt_state)

      model_state = optax.apply_updates(model_state, pipeline_update)
      if debug:
        print(f'Loss on first train batch {loss}')
      
      step += 1
      if step % test_log_freq == 0:
        losses_test = []
        for batch_test in tfds.as_numpy(test_dataset):
          data_test, labels_test = batch_test
          # TODO(gnegiar): build the graphs once before hand, in the dataloading
          graphs_test = tree_utils.tree_stack(
              [make_graph(image, patch_size, bins) for image in data_test])

          loss_test_batch, q = forward(model_state, graphs_test,
                                        graphs_test.sample_start_node_id())
          losses_test.append(loss_test_batch)

          if debug:
            break
        
        loss_test = np.mean(losses_test)
        if loss_test < best_test_loss:
          best_test_loss = loss_test
          print(f"Best test loss: {best_test_loss}")

        if tensorboard_logdir is not None:
          with tf_logger_test.as_default():
            tf.summary.scalar('loss', loss_test, step=step)

          # Save checkpoint
          if loss_test == best_test_loss:
            dir_path = os.path.join(tensorboard_logdir, "extractor_checkpoints")
            os.makedirs(dir_path)
            checkpoint_path = flax.training.checkpoints.save_checkpoint(dir_path, model_state, step=step)
            print(f"Saved checkpoint {checkpoint_path}")

      if debug:
        break
  print('Finished training!')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  local = FLAGS.local
  tensorboard_logdir = FLAGS.tensorboard_logdir
    

  if local:
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    run_id = run_gen.random(8)
    print(f"Run ID: {run_id}")
    tensorboard_logdir = os.path.join(tensorboard_logdir, run_id)

  training_loop(
      optimizer=FLAGS.optimizer,
      patch_size=FLAGS.patch_size,
      batch_size=FLAGS.batch_size,
      dataset=FLAGS.dataset,
      pathfinder_resolution=FLAGS.pathfinder_resolution,
      pathfinder_difficulty=FLAGS.pathfinder_difficulty,
      qkv_dim=FLAGS.qkv_dim,
      mlp_dim=FLAGS.mlp_dim,
      num_heads=FLAGS.num_heads,
      graph_model_hidden_dim=FLAGS.graph_model_hidden_dim,
      learning_rate=FLAGS.learning_rate,
      alpha=FLAGS.alpha,
      rho=FLAGS.rho,
      num_steps_extractor=FLAGS.num_steps_extractor,
      max_subgraph_size=FLAGS.max_subgraph_size,
      max_graph_size=FLAGS.max_graph_size,
      agent_hidden_dim=FLAGS.agent_hidden_dim,
      n_encoder_layers=FLAGS.n_encoder_layers,
      n_epochs=FLAGS.n_epochs,
      ridge=FLAGS.ridge_backward,
      tensorboard_logdir=tensorboard_logdir,
      supernode=FLAGS.supernode,
      overfit=FLAGS.overfit,
      log_freq=FLAGS.log_freq,
      plot_freq=FLAGS.plot_freq,
      test_log_freq=FLAGS.test_log_freq,
      debug=FLAGS.debug,
      seed=FLAGS.seed
  )


if __name__ == '__main__':
  app.run(main)
