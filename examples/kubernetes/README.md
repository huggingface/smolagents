# Kubernetes sandbox

This example demontrate how to set up and use a secure sandbox to run your smolagent in a kubernetes cluster. For the purpose of this example we are using a local cluster installation running minikube

## minikube setup

1. [Install minikube on your system](https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary+download)
2. start minikube and wait it's up and running: `minikube start --memory=10000 --cpus 4`


## Install required python package

Install the Python kubernetes package:
```bash
pip install kubernetes
```
## creating and pushing image to your minikube

[Containerfile](./Containerfile) contains the minimal docker image definition to run a smolagent. You can use it as a starting point to create your own image, adding the package you need

The easiest approach with Minikube is to build the image directly inside Minikube's Docker environment:

1. Point your terminal to use Minikube's Docker daemon:

```bash
eval $(minikube docker-env)
```

2. Build the image directly:

```bash
docker build -t agent-sandbox:latest -f ./Containerfile ./
```
Note: it may take a while



## Running the example in a sandbox

[kubernetes_example.py](./kubernetes_example.py) contain the code to define a Kubernetes sandbox and the example to run a smolagent inside this sandbox

To run it, you just need to call
```bash
python kubernetes_example.py
```


