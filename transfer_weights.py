import paramiko
import os

def download_finetuned_model():
    host = "34.56.24.172"
    username = "hackathon"
    password = "e9f20340"
    
    remote_dir = "/home/hackathon/SwarmGrid/models/finetuned_gemma_warehouse"
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "finetuned_gemma_warehouse"))
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"Created local directory: {local_dir}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}...")
        client.connect(hostname=host, username=username, password=password, timeout=10)
        sftp = client.open_sftp()
        
        print(f"Listing files in {remote_dir}...")
        files = sftp.listdir(remote_dir)
        
        adapter_files = [
            "adapter_config.json", 
            "adapter_model.safetensors", 
            "special_tokens_map.json",
            "tokenizer.json", 
            "tokenizer_config.json", 
            "processor_config.json",
            "preprocessor_config.json",
            "chat_template.jinja",
            "config.json"
        ]
        
        for filename in files:
            if filename not in adapter_files and not filename.startswith("adapter"):
                continue
                
            remote_path = os.path.join(remote_dir, filename).replace('\\', '/')
            local_path = os.path.join(local_dir, filename)
            
            # Simple check if it's a file
            try:
                sftp.stat(remote_path)
                print(f"Downloading {filename}...")
                sftp.get(remote_path, local_path)
            except IOError:
                print(f"Skipping {filename} (not a file or error)")
                
        sftp.close()
        print("\n[SUCCESS] Model transfer complete.")
    finally:
        client.close()

if __name__ == "__main__":
    download_finetuned_model()
