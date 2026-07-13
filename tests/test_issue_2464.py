import time
from smolagents.local_python_executor import timeout, ExecutionTimeoutError

@timeout(1)
def hangs_forever():
    time.sleep(30)    # or any blocking call that exceeds 1 s
    return "done"

hangs_forever()       # freezes the whole process
