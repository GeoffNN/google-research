"""Launch an MNIST experiment."""

from datetime import datetime
from xmanager import xm
from xmanager import xm_local
import itertools
import asyncio

from xmanager.cloud import vertex

import os

def main(*args):
    del args
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_title = f"jaxsel/pathfinder_{date}"

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dockerfile')) as f:
        docker_instructions = f.readlines()

    with xm_local.create_experiment(experiment_title=exp_title) as experiment:
        # Get path of the jaxsel module. 3 directories up from this file for correct requirements.txt
        jaxsel_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spec = xm.PythonContainer(
            path=jaxsel_path,
            entrypoint=xm.ModuleName("jaxsel.examples.train"),
            base_image='gcr.io/deeplearning-platform-release/base-cu113',
            docker_instructions=docker_instructions
        )

        [executable] = experiment.package(
            [
                xm.Packageable(
                    executable_spec=spec,
                    executor_spec=xm_local.Vertex.Spec(),
                ),
            ]
        )

        batch_sizes = [32]
        max_subgraph_sizes = [300] #, 200, 500]
        curiosity_weights = [1., 0.]
        label_weights = [0., 1.]
        entropy_weights = [10.]
        learning_rates = [5e-5]
        alphas = [2e-5]
        rhos = [1e-4]
        ridges = [1e-5]

        resolution = 32
        difficulty = 'hard'
        trials = {}

        trials[32] = list(
            dict(
                [
                    ("dataset", 'lra_pathfinder'),
                    ("batch_size", bs),
                    ("learning_rate", lr),
                    ("max_subgraph_size", max_subgraph_size),
                    ("alpha", alpha),
                    ("rho", rho),
                    ('ridge_backward', ridge),
                    ('n_epochs', 200),
                    ('num_steps_extractor', 30),
                    ('n_encoder_layers', 1),
                    ('num_heads', 2),
                    ('mlp_dim', 32),
                    ('graph_model_hidden_dim', 32), 
                    ('qkv_dim', 16),
                    ('patch_size', 9),
                    ('curiosity_weight', curiosity_weight),
                    ('label_weight', label_weight),
                    ('entropy_weight', entropy_weight),
                    ('pathfinder_resolution', resolution),
                    ('max_graph_size', resolution ** 2 + 10),
                    ('pathfinder_difficulty', difficulty),
                    ('seed', 0),
                ]
            )
            for (bs, lr, max_subgraph_size, alpha, rho, ridge, curiosity_weight, label_weight, entropy_weight) in itertools.product(
                batch_sizes, learning_rates, max_subgraph_sizes, alphas, rhos, ridges, curiosity_weights, label_weights, entropy_weights
            )
        )

        requirements = xm.JobRequirements(A100=1)

        tensorboard = vertex.get_default_client().get_or_create_tensorboard(exp_title)
        tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)

        for i, hyperparameters in enumerate(trials[resolution]):

            output_dir = os.environ.get('GOOGLE_CLOUD_BUCKET_NAME', None)
            output_dir = os.path.join(output_dir, exp_title)

            if output_dir:
                # TODO(gnegiar): Understand how to show all runs on the same TB
                output_dir = os.path.join(output_dir, str(experiment.experiment_id),
                                        str(i)
                                        )
            tensorboard_capability = xm_local.TensorboardCapability(
                name=tensorboard, base_output_directory=output_dir)

            experiment.add(xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements, tensorboard=tensorboard_capability),
                    args=hyperparameters,
                ))


if __name__ == '__main__':
    main()
