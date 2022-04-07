"""Launch an MNIST experiment."""

from xmanager import xm
from xmanager import xm_local
import itertools

import os

def main():
    with xm_local.create_experiment(experiment_title="jaxsel/mnist") as experiment:
        # Get path of the jaxsel module. 3 directories up from this file for correct requirements.txt
        # TODO: Make an examples/requirements and add LRA
        jaxsel_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spec = xm.PythonContainer(
            path=jaxsel_path,
            entrypoint=xm.ModuleName("train"),
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

        for hyperparameters in trials:
            experiment.add(xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements),
                    args=hyperparameters,
                ))


if __name__ == '__main__':
    main()
