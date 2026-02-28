import paramiko

def verify_output():
    host = "34.56.24.172"
    username = "hackathon"
    password = "e9f20340"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, username=username, password=password, timeout=10)
        cmd = "tail -n 20 ~/SwarmGrid/nohup.out"
        stdin, stdout, stderr = client.exec_command(cmd)
        print(stdout.read().decode())
    finally:
        client.close()

if __name__ == "__main__":
    verify_output()
