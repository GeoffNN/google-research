"""Install Jaxsel package."""
import setuptools

setuptools.setup(
    name="jaxsel",
    version="0.0.1",
    license="Apache 2.0",
    install_requires=[
        "jax>=0.3.4",
        "numpy>=1.21",
        "flax>=0.4",
        "matplotlib>=3.5",
        "tensorflow>=2.8",
        "tensorflow-datasets>=4.5",
        "optax>=0.1.1",
        "tree-math>=0.1",
        "lra>=0.0.2",
    ],
    url="https://github.com/google-research/google-research/" "tree/master/jaxsel",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
