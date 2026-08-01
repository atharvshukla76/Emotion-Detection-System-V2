import os
import zipfile

datasets = [
    ("ravdess_features.npz", "unpacked_ravdess"),
    ("samm_features.npz", "unpacked_samm"),
    ("cremad_features.npz", "unpacked_cremad")
]

def unpack_npz(file_path, extract_dir):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, not found.")
        return

    if os.path.exists(extract_dir):
        print(f"Skipping {file_path}, already extracted to {extract_dir}.")
        return

    print(f"Extracting {file_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print(f"Successfully extracted {file_path} to {extract_dir}.")

if __name__ == "__main__":
    print("Starting zero-RAM dataset extraction...")
    for npz_file, out_dir in datasets:
        unpack_npz(npz_file, out_dir)
    print("All datasets unpacked successfully!")
