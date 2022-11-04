python /home/eecs/geoffrey_negiar/google-research/jaxsel/jaxsel/examples/train.py \
  --batch_size 32

max_subgraph_sizes = [300, 500] #, 200, 500]
curiosity_weights = [0., 1.]
learning_rates = [5e-5]
alphas = [2e-5]
rhos = [0, 1e-3]
ridges = [1e-5]