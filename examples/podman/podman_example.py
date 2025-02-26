import podman
import os
from typing import Optional

import podman.errors

class PodmanSandbox:
    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ModuleNotFoundError:
            pass
        print("Initializing Podman sandbox, hold on...")

        # Try different socket URLs
        socket_urls = [
            "unix:///run/podman/podman.sock",  # Default
            "unix:///run/user/{}/podman/podman.sock".format(os.getuid()),  # Rootless
            "http://localhost:8080"  # TCP if configured
        ]
        
        self.client = None
        self.container = None
        connection_error = None
        
        for url in socket_urls:
            try:
                self.client = podman.PodmanClient(base_url=url)
                # Test connection
                self.client.ping()
                print(f"Successfully connected to Podman at {url}")
                break
            except Exception as e:
                connection_error = e
                continue

        
                
        if self.client is None:
            raise RuntimeError(f"Failed to connect to Podman. Tried URLs: {socket_urls}. Last error: {connection_error}")
        
        
    def create_container(self):
        print(f"Current working directory: {os.getcwd()}")
        try:
            if self.client.images.exists("agent-sandbox"):
                print("image already exist, skip creation")
            else:
                self.client.images.build(
                    path="./",
                    dockerfile="./Containerfile",
                    tag="agent-sandbox",
                    rm=True,
                    pull=True,
                    forcerm=True,
                    buildargs={},
                    # decode=True
                )
        except podman.errors.BuildError as e:
            print("Build error logs: ")
            for log in e.build_log:
                if 'stream' in log:
                    print(log['stream'].strip())
            raise

        # Create container with security constraints and proper logging
        self.container = self.client.containers.run(
            image="agent-sandbox",
            detach=True,
            tty=True,
            #security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN")
            },
        )

    def run_code(self, code: str) -> Optional[str]:
        if not self.container:
            self.create_container()

        # Execute code in container
        exec_result = self.container.exec_run(
            cmd=["python", "-c", code],
            user="nobody",
            #demux=True
        )
        
        # Collect all output
        return exec_result[1].decode("utf-8", errors="ignore") if exec_result else None


    def cleanup(self):
        if self.container:
            try:
                self.container.stop()
            except podman.errors.NotFound:
                # Container already removed, this is expected
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.container = None  # Clear the reference

# Example usage:
sandbox = PodmanSandbox()

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
