import paramiko
import os

def deploy_and_restart():
    host = "34.56.24.172"
    username = "hackathon"
    password = "e9f20340"
    
    local_agent_path = os.path.join(os.path.dirname(__file__), "src", "agent.py")
    remote_agent_path = "/home/hackathon/SwarmGrid/src/agent.py"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}...")
        client.connect(hostname=host, username=username, password=password, timeout=10)
        
        # Upload using SFTP
        print(f"Uploading {local_agent_path} to {remote_agent_path}...")
        sftp = client.open_sftp()
        sftp.put(local_agent_path, remote_agent_path)
        sftp.close()
        
        # Restart Uvicorn in the background
        commands = [
            "pkill -f uvicorn",
            "nohup bash -c 'cd ~/SwarmGrid && source ~/.bashrc && export PYTHONPATH=. && export MOCK_AGENT=false && uvicorn src.server:app --host 0.0.0.0 --port 8000' > ~/SwarmGrid/nohup.out 2>&1 &",
            "sleep 5"
        ]
        
        with open('vm_output_utf8.txt', 'w', encoding='utf-8') as f:
            for cmd in commands:
                print(f"Running: {cmd}")
                f.write(f"\n--- Running: {cmd} ---\n")
                stdin, stdout, stderr = client.exec_command(f"bash -i -c '{cmd}'")
                out = stdout.read().decode('utf-8', errors='replace')
                err = stderr.read().decode('utf-8', errors='replace')
                if out: f.write("STDOUT:\n" + out)
                if err: f.write("STDERR:\n" + err)
                
        print("Backend successfully restarted on VM!")
    except Exception as e:
        print(f"Deployment failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    deploy_and_restart()
