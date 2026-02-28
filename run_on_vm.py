import paramiko

def run_ssh_cmd(host, username, password, commands):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, username=username, password=password, timeout=10)
        with open('vm_output_utf8.txt', 'w', encoding='utf-8') as f:
            for cmd in commands:
                f.write(f"\n--- Running: {cmd} ---\n")
                
                # Run the command with pty=False to ensure true detachment
                stdin, stdout, stderr = client.exec_command(f"bash -c '{cmd}'", get_pty=False)
                out = stdout.read().decode('utf-8', errors='replace')
                err = stderr.read().decode('utf-8', errors='replace')
                
                if out: f.write("STDOUT:\n" + out)
                if err: f.write("STDERR:\n" + err)
                
    except Exception as e:
        print(f"Deployment failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    commands = [
        "cat ~/SwarmGrid/nohup.out"
    ]
    run_ssh_cmd("34.56.24.172", "hackathon", "e9f20340", commands)
