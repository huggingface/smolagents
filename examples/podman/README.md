# Podman sandbox

## Podman setup

## Installation

1. [Install Podman on your system](https://podman.io/docs/installation)
2. Install the Python Podman package:
```bash
pip install podman
```
## Setting up Podman image

[Containerfile](./Containerfile) contains the minimal podman image definition to run a smolagent. You can use it as a starting point to create your own image, adding the package you need

## Running the example in a sandbox

[podman_example.py](./podman_example.py) contain the code to define a Podman sandbox and the example to run a smolagent inside this sandbox

To run it, you just need to call
```bash
python podman_example.py
```


