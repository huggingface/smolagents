import socket
import sys
import traceback
import io

exec_globals = {}
exec_locals = {}

def execute_code(code):
    stdout = io.StringIO()
    stderr = io.StringIO()
    sys.stdout = stdout
    sys.stderr = stderr

    try:
        exec(code, exec_globals, exec_locals)
    except Exception:
        traceback.print_exc(file=stderr)
    
    output = stdout.getvalue()
    error = stderr.getvalue()

    # Restore original stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return output + error

def start_server(host='0.0.0.0', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                data = conn.recv(1024)
                if not data:
                    break
                code = data.decode('utf-8')
                output = execute_code(code)
                conn.sendall(output.encode('utf-8'))

if __name__ == "__main__":
    start_server() 