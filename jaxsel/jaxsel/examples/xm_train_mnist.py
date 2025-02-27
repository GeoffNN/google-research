"""Launch an MNIST experiment."""

from xmanager import xm
from xmanager import xm_local
import itertools
import asyncio

from xmanager.cloud import vertex

import os

def main(*args):
    del args
    exp_title = "jaxsel/mnist"

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dockerfile')) as f:
        docker_instructions = f.readlines()

    with xm_local.create_experiment(experiment_title=exp_title) as experiment:
        # Get path of the jaxsel module. 3 directories up from this file for correct requirements.txt
        # TODO: Make an examples/requirements and add LRA
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

        # batch_sizes = [64] 
        # max_subgraph_sizes = [200] #, 200, 500]
        # learning_rates = [5e-3, 3e-3, 1e-3]
        # alphas = [2e-4]
        # rhos = [0., 1e-5, 1e-6]

        # TODO(gnegiar): Large batch sizes

        batch_sizes = [64] 
        max_subgraph_sizes = [200] #, 200, 500]
        learning_rates = [1e-3]
        curiosity_weights = [1.]
        entropy_weights = [1e1]
        label_weights = [1.]
        alphas = [2e-4]
        rhos = [1e-4, 0.]
        ridges = [1e-7]

        trials = list(
            dict(
                [
                    ("batch_size", bs),
                    ("learning_rate", lr),
                    ("max_subgraph_size", max_subgraph_size),
                    ("alpha", alpha),
                    ("rho", rho),
                    ('n_epochs', 100),
                    ('ridge_backward', ridge),
                    ('curiosity_weight', curiosity_weight),
                    ('entropy_weight', entropy_weight),
                    ('label_weight', label_weight),
                    ('exploration_steps', 400)
                ]
            )
            for (bs, lr, max_subgraph_size, alpha, rho, ridge,curiosity_weight, label_weight, entropy_weight) in itertools.product(
                batch_sizes, learning_rates, max_subgraph_sizes, alphas, rhos, ridges, curiosity_weights, label_weights, entropy_weights
            )
        )

        requirements = xm.JobRequirements(A100=1)

        tensorboard = vertex.get_default_client().get_or_create_tensorboard(exp_title)
        tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)

        for i, hyperparameters in enumerate(trials):

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
