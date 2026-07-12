from smolagents.local_python_executor import LocalPythonExecutor

executor = LocalPythonExecutor(additional_authorized_imports=["threading"])
result = executor("""
import threading
lock = threading.Lock()
with lock:
    x = 1
x
""")
