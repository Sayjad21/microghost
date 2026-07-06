import os
from huggingface_hub import HfFileSystem, hf_hub_download
from concurrent.futures import ThreadPoolExecutor

# Replace this with your token or set it as an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_TOKEN")

def download_subset(repo_id, local_dir, max_files=50, file_extensions=('.jpg', '.png', '.txt', '.json', '.xml')):
    print(f"Fetching file list for {repo_id}...")
    fs = HfFileSystem()
    
    # List files in the root or specific folders. We'll do a recursive glob but stop early if possible
    # Actually, fs.ls can be slow if there are 96000 files in one flat directory.
    # Let's try to get a generator of files and stop early.
    
    # HF dataset structure usually contains data in some folders or zip files.
    # Let's list the root first.
    root_files = fs.ls(f"datasets/{repo_id}", detail=False)
    
    # We want to find images and annotations. Let's traverse down a bit.
    files_to_download = []
    
    # A simple DFS to find files
    stack = [f"datasets/{repo_id}"]
    
    while stack and len(files_to_download) < max_files:
        current_dir = stack.pop()
        try:
            items = fs.ls(current_dir, detail=True)
            for item in items:
                if item['type'] == 'directory':
                    stack.append(item['name'])
                else:
                    name = item['name']
                    if any(name.lower().endswith(ext) for ext in file_extensions):
                        files_to_download.append(name.replace(f"datasets/{repo_id}/", ""))
                        if len(files_to_download) >= max_files:
                            break
        except Exception as e:
            print(f"Error reading {current_dir}: {e}")
            
    print(f"Found {len(files_to_download)} files to download from {repo_id}.")
    
    os.makedirs(local_dir, exist_ok=True)
    
    def download_file(filepath):
        try:
            hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filepath, local_dir=local_dir)
            print(f"Downloaded {filepath}")
        except Exception as e:
            print(f"Failed to download {filepath}: {e}")
            
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_file, files_to_download)

if __name__ == "__main__":
    base_dir = r"c:\Users\Legion\OneDrive\Desktop\ieee\data"
    os.makedirs(base_dir, exist_ok=True)
    
    download_subset("etri/ForestPersons", os.path.join(base_dir, "ForestPersons"), max_files=5000)
    download_subset("etri/ForestPersonsIR", os.path.join(base_dir, "ForestPersonsIR"), max_files=5000)
    print("Done!")
