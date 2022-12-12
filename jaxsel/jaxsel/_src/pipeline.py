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

"""Provides a Pipeline Flax module, wrapping the whole forward pass."""

import functools
from typing import Tuple

from flax import struct
import flax.linen as nn
import jax
from jax.experimental import sparse as jsparse
import jax.numpy as jnp

from jaxsel._src import graph_api
from jaxsel._src import graph_models
from jaxsel._src import losses
from jaxsel._src import subgraph_extractors


@struct.dataclass
class ClassificationPipelineConfig:
  """Contains the config for a classification pipeline.

  Attributes:
    extractor_config: Config for subgraph extractor.
    graph_classifier_config: Config for graph classifier.
    use_node_weights: Whether to use the learned node weights, or only the
      support.
    supernode: whether to use a supernode in the extracted subgraph. Attempt to
      mitigate range issues with large subgraphs.
  """
  extractor_config: subgraph_extractors.ExtractorConfig
  graph_classifier_config: graph_models.TransformerConfig
  supernode: bool = False
  use_node_weights: bool = True
  curiosity_weight: float = 0.
  entropy_weight: float = 0.
  label_weight: float = 1.


class ClassificationPipeline(nn.Module):
  """Flax module for performing inference on a graph."""

  config: ClassificationPipelineConfig

  def setup(self):
    self.extractor = subgraph_extractors.SparseISTAExtractor(
        config=self.config.extractor_config)
    self.graph_classifier = graph_models.SimpleGraphClassifier(
        config=self.config.graph_classifier_config)

  def _add_supernode(
      self, node_features, dense_submat,
      dense_q):
    """Adds supernode with full incoming and outgoing connectivity.

    Adds a row and column of 1s to `dense_submat`, and normalizes. Also adds a
      row to `node_features`, containing the average of the other node features.
      Adds a weight of 1 at the end of `dense_q`.

    Args:
      node_features: Shape (num_nodes, feature_dim) Matrix of node features.
      dense_submat: Shape (num_nodes, num_nodes) Adjacency matrix.
      dense_q: Shape (num_nodes,) Node weights.

    Returns:
      node_features: Shape (num_nodes + 1, feature_dim) Matrix of node features.
      dense_submat: Shape (num_nodes + 1, num_nodes + 1) Adjacency matrix.
      dense_q: Shape (num_nodes + 1,) Node weights.
    """
    dense_submat = jnp.row_stack(
        (dense_submat, jnp.ones(dense_submat.shape[1])))
    dense_submat = jnp.column_stack(
        (dense_submat, jnp.ones(dense_submat.shape[0])))
    # Normalize nonzero elements
    # The sum is bounded away from 0, so this is always differentiable
    # TODO(gnegiar): Do we want this? It means the supernode gets half the
    # outgoing weights
    dense_submat = dense_submat / dense_submat.sum(axis=-1, keepdims=True)
    # Add a weight to the supernode
    dense_q = jnp.append(dense_q, jnp.mean(dense_q))
    # We embed the supernode using a distinct value.
    # TODO(gnegiar): Should we use another embedding?
    node_features = jnp.append(
        node_features,
        jnp.full((1, node_features.shape[1]), 2, dtype=int),
        axis=0)
    return node_features, dense_submat, dense_q

  def loss_fun(self, logits, label):
    """Computes the classification loss."""
    num_classes = self.config.graph_classifier_config.num_classes
    if num_classes > 2:
      return losses.cross_entropy_loss(logits, label, num_classes=num_classes)
    else:
      return losses.binary_logistic_loss(logits, label, num_classes=num_classes)

  def curiosity_loss_fun(self, image, q, node_ids, graphs):
    """Computes the curiosity loss function.
    
    This loss function encourages exploration, similarly to 
    exploration bonuses in RL.
    since we are considering black & white image problems, 
    we start by encouraging selection of fore-ground pixels.
    """
    # Extract values of the image at the extracted pixels
    extracted_image = jnp.take(jnp.clip(image.flatten(), a_min=0.), node_ids, fill_value=0.)
    # Normalize the extracted image and the extracted node weights to match scale
    normalized_image = jax.nn.softmax(jnp.where(extracted_image!=0, extracted_image, -jnp.inf))
    normalized_q = jax.nn.softmax(jnp.where(q!=0, q, -jnp.inf))

    # We reweight the matching loss by the inverse distance from the start node: we want to
    # encourage going far from the start node
    pixel_coords = jnp.mgrid[0:image.shape[0]:1,0:image.shape[1]:1].transpose(1, 2, 0)
    start_pixel_coords = graphs.start_node_coords[None, None] # Reshape to match image shape
    # TODO: use norm squared?
    distance_from_start_node = jnp.linalg.norm(pixel_coords - start_pixel_coords, axis=-1)
    extracted_reweighting = jnp.take(distance_from_start_node, node_ids, fill_value=0.)

    loss = ((normalized_q - normalized_image) ** 2 * extracted_reweighting).sum()
    return loss

  def entropy_loss_fun(self, q):
    """Computes entropy loss."""
    # Normalize q so it's a probability, and compute entropy
    entropy = jax.scipy.special.entr(q / sum(q)).sum()
    max_entropy = -jnp.log(1 / q.shape[0])  # Uniform weights on the support of q
    entropy_loss = max_entropy - entropy
    return entropy_loss


  def pred_fun(self, logits):
    """Computes the model's predicted class."""
    num_classes = self.config.graph_classifier_config.num_classes
    if num_classes > 2:
      return logits.argmax(-1)
    else:
      return (jax.scipy.special.expit(logits) > .5).astype(int).squeeze()

  def __call__(
      self, graph,
      start_node_id):
    """Predicts class logits for a given graph.

    Args:
      graph: The graph to process.
      start_node_id: The graph start node.

    Returns:
      logits: class logits
      (q, dense_submat): node weights and learned edge weights,
        for visualization purposes.
    """
    (dense_q, node_features, node_ids, dense_submat, q, adjacency_matrix,
     error) = self.extractor(start_node_id, graph)

    del adjacency_matrix
    del error

    if not self.config.use_node_weights:
      # Don't use the weighting.
      dense_q = jnp.where(dense_q != 0, 1., 0.)
    if self.config.supernode:
      node_features, dense_submat, dense_q = self._add_supernode(
          node_features, dense_submat, dense_q)
    # TODO(gnegiar): Clarify graph models arguments:
    # should we take the learned adjacency matrix or not?
    # Should we pass the node weights through a nonlinearity?
    # Commented lines below reflect these WIP possibilities.
    logits = self.graph_classifier(
        node_features=node_features[..., :-1], # Remove node_ids from node features.
        node_ids=node_ids,
        adjacency_mat=jnp.where(dense_submat[Ellipsis, jnp.newaxis] != 0, 1., 0.),  # TODO: Try without where using the true values?
        # adjacency_mat=dense_submat[..., jnp.newaxis],
        qstar=dense_q / (dense_q.std() + 1e-8),  # Normalizing for scale: this seems to work best
        # qstar=10 * (dense_q**2)**(1. / 8),  # normalizing for scale
    )

    return logits, (q, dense_submat, node_ids, dense_q)

  @functools.partial(
      nn.vmap,
      in_axes=(0, 0, 0),
      variable_axes={'params': None},
      split_rngs={
          'params': False,
          'dropout': True
      })
  def compute_loss(self, graphs, start_node_ids,
                   labels):
    """Computes the loss encurred by our model over a batch.

    Args:
      graphs: Contains the underlying unweighted graphs. All pytree node leaves
        expect a batch dimension.
      start_node_ids: Where to start on the graphs. Expects a batch dimension.
      labels: True labels for the examples. Expects a batch dimension.

    Returns:
      loss_value: value of the loss for the given graph.
    """
    # Extract subgraph and associated features
    cfg = self.config
    logits, (q, dense_submat, node_ids, dense_q) = self(graphs, start_node_ids)
    preds = self.pred_fun(logits)
    label_loss = self.loss_fun(logits, labels)
    curiosity_loss = self.curiosity_loss_fun(graphs.image, dense_q, node_ids, graphs) if cfg.curiosity_weight > 0. else 0.
    # Add entropy term to deconcentrate the weights
    entropy_loss = self.entropy_loss_fun(dense_q)
    loss_vals = cfg.label_weight * label_loss + cfg.curiosity_weight * curiosity_loss + cfg.entropy_weight * entropy_loss
    return loss_vals, (preds, logits, q, dense_submat, label_loss, curiosity_loss, entropy_loss)
