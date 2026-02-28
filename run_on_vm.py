import paramiko
import os
import time

def deploy_video_feature():
    host = "34.56.24.172"
    username = "hackathon"
    password = "e9f20340"
    
    files_to_upload = [
        ("src/server.py", "/home/hackathon/SwarmGrid/src/server.py"),
        ("src/index.html", "/home/hackathon/SwarmGrid/src/index.html")
    ]
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}...")
        client.connect(hostname=host, username=username, password=password, timeout=10)
        
        sftp = client.open_sftp()
        for local_rel, remote_abs in files_to_upload:
            local_abs = os.path.join(os.path.dirname(__file__), local_rel)
            print(f"Uploading {local_rel}...")
            sftp.put(local_abs, remote_abs)
        sftp.close()
        
        print("Force killing old processes...")
        client.exec_command("pkill -9 -f uvicorn")
        time.sleep(2)
        
        print("Restarting backend (Wait 25s for BF16 load)...")
        cmd = "cd ~/SwarmGrid && nohup python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8000 > nohup.out 2>&1 &"
        client.exec_command(cmd)
        
        time.sleep(25)
        
        print("Verifying port 8000...")
        stdin, stdout, stderr = client.exec_command("ss -tulpn | grep 8000")
        res = stdout.read().decode()
        if "8000" in res:
            print(f"Success! Port 8000 is listening.")
        else:
            print("Failed to bind to port 8000.")
            
    finally:
        client.close()

if __name__ == "__main__":
    deploy_video_feature()
