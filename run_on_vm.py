import paramiko

def find_models():
    host = "34.56.24.172"
    username = "hackathon"
    password = "e9f20340"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, username=username, password=password, timeout=10)
        # Check env file on VM first to see what it was using
        print("--- VM .env configuration ---")
        stdin, stdout, stderr = client.exec_command("cat ~/SwarmGrid/.env | grep 'MODEL\|MODEL_PATH'")
        print(stdout.read().decode())
        
        # Search for common model directories
        print("--- Searching for weight directories ---")
        cmd = "find ~/SwarmGrid -name 'config.json' -not -path '*/.*'"
        stdin, stdout, stderr = client.exec_command(cmd)
        print(stdout.read().decode())
    finally:
        client.close()

if __name__ == "__main__":
    find_models()
