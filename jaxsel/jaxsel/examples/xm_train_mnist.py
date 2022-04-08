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
    with xm_local.create_experiment(experiment_title=exp_title) as experiment:
        # Get path of the jaxsel module. 3 directories up from this file for correct requirements.txt
        # TODO: Make an examples/requirements and add LRA
        jaxsel_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spec = xm.PythonContainer(
            path=jaxsel_path,
            entrypoint=xm.ModuleName("jaxsel.examples.train"),
        )

        [executable] = experiment.package(
            [
                xm.Packageable(
                    executable_spec=spec,
                    executor_spec=xm_local.Vertex.Spec(),
                ),
            ]
        )

        batch_sizes = [32, 64, 1024]
        max_subgraph_sizes = [100 ] #, 200, 500]
        learning_rates = [0.1, 0.001]

        # TODO: Add fixed hyperparams here.
        # common_args = {}
        # TODO: add tensorboard logdir

        trials = list(
            dict(
                [
                    ("batch_size", bs),
                    ("learning_rate", lr),
                    ("max_subgraph_size", max_subgraph_size),
                ]
            )
            for (bs, lr, max_subgraph_size) in itertools.product(
                batch_sizes, learning_rates, max_subgraph_sizes
            )
        )

        requirements = xm.JobRequirements(A100=1)

        tensorboard = vertex.get_default_client().get_or_create_tensorboard(exp_title)
        tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)

        for i, hyperparameters in enumerate(trials):

            output_dir = os.environ.get('GOOGLE_CLOUD_BUCKET_NAME', None)
            output_dir = os.path.join(output_dir, exp_title)

            if output_dir:
                output_dir = os.path.join(output_dir, str(experiment.experiment_id),
                                        str(i))
            tensorboard_capability = xm_local.TensorboardCapability(
                name=tensorboard, base_output_directory=output_dir)

            experiment.add(xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements, tensorboard=tensorboard_capability),
                    args=hyperparameters,
                ))


if __name__ == '__main__':
    main()
