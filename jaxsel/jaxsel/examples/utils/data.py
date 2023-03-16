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

"""Contains utilities for dataloading."""

import functools
from typing import Tuple, Sequence

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage as ndi
import tensorflow as tf
import tensorflow_datasets as tfds

from jaxsel._src import image_graph
from jaxsel.examples.utils import pathfinder_data



def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def load_mnist(
    batch_size = 64
):
  """Load MNIST train and test datasets into memory.

  Taken from https://github.com/google/flax/blob/main/examples/mnist/train.py.

  Args:
    batch_size: batch size for both train and test.

  Returns:
    train_dataset, test_dataset, image_shape, num_classes
  """
  train_dataset, test_dataset = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

  train_dataset = train_dataset.map(
      normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  test_dataset = test_dataset.map(
      normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  train_dataset = train_dataset.cache()

  train_dataset = train_dataset.shuffle(
      60_000, seed=0, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

  test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
  test_dataset = test_dataset.cache()
  return train_dataset, test_dataset, (28, 28), 10


def load_pathfinder(
    batch_size = 64,
    resolution = 32,
    difficulty = 'easy',
    overfit = False
):
  """Loads the pathfinder data.

  Args:
    batch_size: batch size for train, test and val datasets.
    resolution: resolution of the task. Can be 32, 64 or 128.
    difficulty: difficulty of the task, defined by the number of distractor
      paths. Must be in ['easy', 'intermediate', 'hard'].
    overfit: if True, the datasets are all the same: first 2 samples of the
      validation dataset.

  Returns:
    train_dataset, val_dataset, test_dataset, image_shape, num_classes
  """
  (train_dataset, val_dataset, test_dataset, num_classes, vocab_size,
   image_shape) = pathfinder_data.load(
       n_devices=1,
       batch_size=batch_size,
       resolution=resolution,
       normalize=True,  # Normalize to 0, 1
       difficulty=difficulty)

  del vocab_size

  if overfit:
    # Doesn't use batch_size in this case
    n_overfit = 8
    train_dataset = val_dataset.unbatch().take(n_overfit).batch(n_overfit)
    val_dataset = train_dataset
    test_dataset = train_dataset

  # Make datasets returns tuples of images, labels
  def tupleize(datapoint):
    return tf.cast(datapoint['inputs'], tf.float32), datapoint['targets']

  train_dataset = train_dataset.map(
      tupleize, num_parallel_calls=tf.data.AUTOTUNE)
  val_dataset = val_dataset.map(
      tupleize, num_parallel_calls=tf.data.AUTOTUNE)
  test_dataset = test_dataset.map(
      tupleize, num_parallel_calls=tf.data.AUTOTUNE)

  return train_dataset, val_dataset, test_dataset, image_shape, num_classes


# TODO(gnegiar): Map this on the dataset, and cache it.
@functools.partial(jax.jit, static_argnums=(1,))
def make_graph_mnist(
    image, patch_size, bins = np.array([0., .3, 1.])
):
  """Makes a graph object to hold an MNIST sample.

  Args:
    image: Should be squeezable to a 2d array
    patch_size: size of patches for node features.
    bins: Used for binning the pixel values. The highest bin must be greater
      than the highest value in image.

  Returns:
    graph representing the image.
  """
  return image_graph.ImageGraph.create(
      # The threshold value .3 was selected to keep information
      # while not introducing noise
      jnp.digitize(image, bins).squeeze(),
      get_start_pixel_fn=lambda _: (14, 14),  # start in the center
      num_colors=len(bins),  # number of bins + 'out of bounds' pixel
      patch_size=patch_size)


# TODO(gnegiar): Allow multiple start nodes.
def _get_start_pixel_fn_pathfinder(
    image, nbins, thresh = .5):
  """Detects a probable start point in a Pathfinder image example."""
  thresh_image = jnp.where(image > thresh * nbins, 1, 0)
  distance = ndi.distance_transform_edt(thresh_image)
  idx = distance.argmax()
  coords = np.unravel_index(idx, thresh_image.shape)
  return coords

def _get_end_pixel_fn_pathfinder(
    image, nbins, thresh = .5):
  """Detects a probable start point in a Pathfinder image example."""
  thresh_image = jnp.where(image > thresh * nbins, 1, 0)
  distance = ndi.distance_transform_edt(thresh_image)
  idx = np.argpartition(distance.flatten(), -1)[-1]
  coords = np.unravel_index(idx, thresh_image.shape)
  return coords

  
def cut_patch(image, coords, h):
    """Returns a patch of size 2h+1 around coords from image."""
    x, y = coords
    patch = image[x-h:x+h+1, y-h:y+h+1].copy()
    image[x-h:x+h+1, y-h:y+h+1] = 0
    return patch

def clip_coords(coords, h, shape):
    """Clips coords so that the h-sized patch around coords is within shape."""
    x, y = coords
    x = np.clip(x, h, shape[0] - h - 2)
    y = np.clip(y, h, shape[1] - h - 2)
    return x, y

def move_endpoint_to_random_location_near_startpoint(image, start_coord, end_coords, h, max_dist=20):
    """Moves the endpoint patch to a random location in the image."""
    # Sample random location near startpoint
    dist = max_dist + 1
    while dist > max_dist or dist < 3:
        random_flat_idx = np.random.randint(0, image.shape[0] * image.shape[1])
        rnd_coords = np.unravel_index(random_flat_idx, image.shape)
        rnd_coords = clip_coords(rnd_coords, h, image.shape)
        dist = np.linalg.norm(np.array(start_coord) - np.array(rnd_coords), ord=1)

    # Cut patch 
    image_copy = image.copy()
    end_coords = clip_coords(end_coords, h, image.shape)
    patch = cut_patch(image_copy, end_coords, h)
    
    # and paste it at random location
    image_copy[rnd_coords[0] -h: rnd_coords[0] + h+1, rnd_coords[1] - h: rnd_coords[1] + h+1] = patch
    return image_copy, rnd_coords

def create_random_line_between_points(start_point, end_point, shape):
    """
    Creates a dashed line between start_point and end_point.
    """
    # Sample the start point of the curve
    points = []
    start_point, end_point = map(np.array, (start_point, end_point))
    cur_point = start_point.astype(float)
    momentum = 2 * np.random.randn(2)

    step = 1
    while not np.all(np.abs(cur_point - end_point) < 1):
      # Use the same grad for multiple steps to make the line flatter?
      grad = np.sign(end_point - cur_point)
      momentum = .8 * momentum + .2 * grad / np.sqrt(step)
      momentum_norm = np.sum(np.abs(momentum))
      cur_point = cur_point + momentum / (1.2 * momentum_norm)

      # clip to image
      cur_point = np.clip(cur_point, 0, np.array(shape) - 1)

      points.append(np.round(cur_point).astype(int))
      step += 1
    return np.stack(points)

def add_dashed_line_to_image(image, points, dash_length):
    """Adds a dashed line to image following the given point coordinates."""
    image_copy = image.copy()
    for k, point in enumerate(points):
        if k % (2 * dash_length) < dash_length:
            xs = np.arange(np.clip(point[0] - 1, 0, image.shape[0] - 1),
                       np.clip(point[0] + 1, 0, image.shape[0]))
            ys = np.arange(np.clip(point[1] - 1, 0, image.shape[1] - 1),
                        np.clip(point[1] + 1, 0, image.shape[1]))
            # Debug case when xs or ys are empty
            if len(ys) == 0 or len(xs) == 0:
                print(point)
            image_copy[min(xs):max(xs) + 1, min(ys):max(ys) + 1] = np.random.uniform(0.5, 0.8, size=(len(xs), len(ys))
              ).reshape(
                image_copy[min(xs):max(xs) + 1, min(ys):max(ys) + 1].shape)
    return image_copy


def batch_with_random_endpoint(data, labels, bins, h=3, max_dist=20, probability=0.5, thresh=0.5):
    """Augments a batch of images with a random endpoint with given probability."""
    images = []
    new_labels = []
    nbins = len(bins)
    for i in range(data.shape[0]):
        if np.random.rand() > probability:
          images.append(data[i].squeeze())
          new_labels.append(labels[i])
        else:
          binned_image = np.digitize(data[i], bins)
          start_coords = _get_start_pixel_fn_pathfinder(binned_image.squeeze(), nbins, thresh=thresh)
          end_coords = _get_end_pixel_fn_pathfinder(binned_image.squeeze(), nbins, thresh) 
          image_with_new_endpoints, new_end_coords = move_endpoint_to_random_location_near_startpoint(data[i].squeeze(), start_coords, end_coords, h, max_dist=max_dist)

          if labels[i] == 1:
            # Add a dashed line between the start and end points
            points_on_line = create_random_line_between_points(start_coords, new_end_coords, data[i].squeeze().shape)
            # TODO: Figure out how dash_length changes with resolution
            image_with_new_endpoints = add_dashed_line_to_image(image_with_new_endpoints, points_on_line, dash_length=8)
            # save image to a new file
            # increment the file name
          images.append(image_with_new_endpoints)
             
    return np.stack(images)[..., None], np.array(labels)


# TODO(gnegiar): Map this on the dataset, and cache it.
def make_graph_pathfinder(
    image,
    patch_size,
    bins,
):
  """Makes a graph holding a pathfinder image.

  Args:
    image: Should be squeezable to a 2d array
    patch_size: size of patches for node features.
    bins: Used for binning the pixel values. The highest bin must be greater
      than the highest value in image.

  Returns:
    graph representing the image.
  """

  # TODO(gnegiar): Allow continuous features in models.
  return image_graph.ImageGraph.create(
      jnp.digitize(image, bins).squeeze(),
      # Set thresh to .5 by leveraging the image discretization.
      get_start_pixel_fn=functools.partial(
          _get_start_pixel_fn_pathfinder, nbins=len(bins), thresh=.5),
      num_colors=len(bins),  # number of bins + 'out of bounds' pixel
      patch_size=patch_size)
