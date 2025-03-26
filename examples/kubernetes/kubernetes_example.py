import os
import time
from typing import Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class KubernetesSandbox:
    def __init__(self, namespace="default"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ModuleNotFoundError:
            pass
        print("Initializing Kubernetes sandbox, hold on...")

        # Try to load Kubernetes config from different locations
        try:
            # Try in-cluster config first (if running inside K8s)
            config.load_incluster_config()
            print("Using in-cluster Kubernetes configuration")
        except config.ConfigException:
            try:
                # Fall back to local kubeconfig file
                config.load_kube_config()
                print("Using local Kubernetes configuration file")
            except config.ConfigException as e:
                raise RuntimeError(f"Failed to connect to Kubernetes: {e}")

        # Initialize Kubernetes API clients
        self.core_api = client.CoreV1Api()
        self.apps_api = client.AppsV1Api()
        self.batch_api = client.BatchV1Api()
        
        # Set the namespace
        self.namespace = namespace
        
        # Container references
        self.pod_name = None
        self.job_name = None
        
        # Check connection to the cluster
        try:
            self.core_api.list_namespace()
            print(f"Successfully connected to Kubernetes cluster")
        except ApiException as e:
            raise RuntimeError(f"Failed to connect to Kubernetes: {e}")
    
    def create_container(self):
        print(f"Current working directory: {os.getcwd()}")
        
        # Generate unique name for this execution
        import uuid
        run_id = str(uuid.uuid4())[:8]
        self.job_name = f"agent-sandbox-{run_id}"
        
        # Ensure the image is available (note: K8s won't build images)
        # You would need to build and push the image to a registry first
        # This example assumes the image is already in a registry
        image_name = "agent-sandbox:latest"
        print(f"Using existing image: {image_name}")
        
        # Create a job to run the container
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=self.job_name,
                namespace=self.namespace
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "agent-sandbox"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="agent-sandbox",
                                image=image_name,
                                image_pull_policy="Never", #it uses local built image directly inside Minikube's Docker environment
                                env=[
                                    client.V1EnvVar(
                                        name="HF_TOKEN",
                                        value=os.getenv("HF_TOKEN")
                                    )
                                ],
                                security_context=client.V1SecurityContext(
                                    run_as_non_root=True,
                                    run_as_user=65534,  # nobody user
                                    capabilities=client.V1Capabilities(
                                        drop=["ALL"]
                                    ),
                                    allow_privilege_escalation=False
                                ),
                                # Keep the container running
                                command=["sleep", "infinity"]
                            )
                        ],
                        restart_policy="Never"
                    )
                ),
                backoff_limit=0  # Don't retry on failure
            )
        )
        
        try:
            self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )
            print(f"Created job: {self.job_name}")
        except ApiException as e:
            raise RuntimeError(f"Failed to create Kubernetes job: {e}")
            
        # Wait for the pod to be ready
        time.sleep(2)  # Give K8s a moment to create the pod
        
        # Find the pod created by the job
        try:
            pods = self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={self.job_name}"
            )
            if not pods.items:
                raise RuntimeError(f"No pods found for job {self.job_name}")
            
            self.pod_name = pods.items[0].metadata.name
            print(f"Pod name: {self.pod_name}")
            
            # Wait for pod to be running
            wait_count = 0
            while wait_count < 30:  # Wait up to 30 seconds
                pod_status = self.core_api.read_namespaced_pod_status(
                    name=self.pod_name,
                    namespace=self.namespace
                )
                if pod_status.status.phase == "Running":
                    break
                if pod_status.status.phase in ["Failed", "Unknown"]:
                    raise RuntimeError(f"Pod {self.pod_name} failed to start")
                print(f"Waiting for pod to start... (status: {pod_status.status.phase})")
                time.sleep(1)
                wait_count += 1
                
            if wait_count >= 30:
                raise RuntimeError(f"Timeout waiting for pod {self.pod_name} to start")
                
        except ApiException as e:
            raise RuntimeError(f"Failed to get pod information: {e}")


    def run_code(self, code: str) -> Optional[str]:
        if not self.pod_name:
            self.create_container()

        import kubernetes.stream as stream
        
        try:
            # Create a direct stream using the stream utility
            exec_stream = stream.stream(
                self.core_api.connect_get_namespaced_pod_exec,
                name=self.pod_name,
                namespace=self.namespace,
                container="agent-sandbox",
                command=["python", "-c", code],
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False  # Important: don't preload content
            )
            
            # Collect the output
            result = ""
            while exec_stream.is_open():
                exec_stream.update(timeout=1)
                if exec_stream.peek_stdout():
                    result += exec_stream.read_stdout()
                if exec_stream.peek_stderr():
                    result += exec_stream.read_stderr()
                    
            exec_stream.close()
            return result
            
        except Exception as e:
            return f"Error executing code: {e}"
    
    def cleanup(self):
        if self.job_name:
            try:
                # Delete the job (and associated pods)
                self.batch_api.delete_namespaced_job(
                    name=self.job_name,
                    namespace=self.namespace,
                    body=client.V1DeleteOptions(
                        propagation_policy="Foreground"
                    )
                )
                print(f"Deleted job: {self.job_name}")
            except ApiException as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.job_name = None
                self.pod_name = None


# Example usage:
if __name__ == "__main__":
    sandbox = KubernetesSandbox()

    try:
        # Define your agent code
        agent_code = """
import os
from smolagents import CodeAgent, HfApiModel

# Initialize the agent
agent = CodeAgent(
    model=HfApiModel(token=os.getenv("HF_TOKEN"), provider="together"),
    tools=[]
)

# Run the agent
response = agent.run("What's the 20th Fibonacci number?")
print(response)
"""

        # Run the code in the sandbox
        output = sandbox.run_code(agent_code)
        print(output)

    finally:
        sandbox.cleanup()