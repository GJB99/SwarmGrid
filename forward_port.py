import paramiko
import select
import socketserver
import threading
import sys

class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            chan = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.chain_host, self.chain_port),
                self.request.getpeername(),
            )
        except Exception as e:
            print(f"Incoming request to {self.chain_host}:{self.chain_port} failed: {e}")
            return
        if chan is None:
            print(f"Incoming request to {self.chain_host}:{self.chain_port} was rejected")
            return

        peer = self.request.getpeername()
        print(f"Connected!  Tunnel open {peer} -> {self.chain_host}:{self.chain_port}")
        while True:
            r, w, x = select.select([self.request, chan], [], [])
            if self.request in r:
                data = self.request.recv(1024)
                if len(data) == 0:
                    break
                chan.send(data)
            if chan in r:
                data = chan.recv(1024)
                if len(data) == 0:
                    break
                self.request.send(data)
        print(f"Tunnel closed from {peer}")

def forward_tunnel(local_port, remote_host, remote_port, transport):
    class SubHander(Handler):
        chain_host = remote_host
        chain_port = remote_port
        ssh_transport = transport
    try:
        server = ForwardServer(("", local_port), SubHander)
        print(f"Now forwarding port {local_port} to {remote_host}:{remote_port} ...")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Canceled by user.")
    except Exception as e:
        print(f"Port forwarding error: {e}")
        sys.exit(1)

def main():
    vm_ip = "34.56.24.172"
    vm_user = "hackathon"
    vm_pass = "e9f20340"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"Connecting to ssh host {vm_ip} ...")
    try:
        client.connect(
            hostname=vm_ip,
            username=vm_user,
            password=vm_pass,
            timeout=10
        )
    except Exception as e:
        print(f"Failed to connect to {vm_ip}: {e}")
        sys.exit(1)

    print("Now forwarding local port 8000 to remote localhost:8000")
    try:
        forward_tunnel(8000, "127.0.0.1", 8000, client.get_transport())
    except KeyboardInterrupt:
        print("Port forwarding stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
