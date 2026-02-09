import os
import shutil
import glob

print("--- cuDNN Installation Helper ---")

# Find cuDNN zip or extracted folder
print("\nSearching for cuDNN...")

# Search in Downloads
downloads = os.path.join(os.path.expanduser("~"), "Downloads")
cudnn_sources = []

# Check for extracted folders
for item in os.listdir(downloads):
    path = os.path.join(downloads, item)
    if os.path.isdir(item) and 'cudnn' in item.lower():
        cudnn_sources.append(path)

# Also check common extraction locations
common_locations = [
    os.path.join(os.path.expanduser("~"), "Downloads"),
    os.path.join(os.path.expanduser("~"), "Desktop"),
]

if not cudnn_sources:
    print("\nCould not find cuDNN automatically.")
    print("\nPlease tell me: Where did you extract the cuDNN zip file?")
    print("(Enter the full path to the extracted cuDNN folder)")
    cudnn_path = input("> ").strip()

    if not os.path.exists(cudnn_path):
        print(f"Error: Path does not exist: {cudnn_path}")
        exit(1)
else:
    cudnn_path = cudnn_sources[0]
    print(f"\nFound cuDNN at: {cudnn_path}")

# CUDA 12.4 destination
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cuda_include = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include"
cuda_lib = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64"

print(f"\nTarget CUDA 12.4 folders:")
print(f"  bin: {cuda_bin}")
print(f"  include: {cuda_include}")
print(f"  lib: {cuda_lib}")

# Find and copy files
print("\nCopying cuDNN files...")

# Look for bin, include, lib folders in cudnn_path
for folder in ['bin', 'include', 'lib']:
    src = os.path.join(cudnn_path, folder)
    if not os.path.exists(src):
        # Try with cuda in path
        src = os.path.join(cudnn_path, 'cuda', folder)

    if os.path.exists(src):
        if folder == 'bin':
            dest = cuda_bin
            files = glob.glob(os.path.join(src, '*.dll'))
        elif folder == 'include':
            dest = cuda_include
            files = glob.glob(os.path.join(src, '*.h'))
        elif folder == 'lib':
            dest = cuda_lib
            files = glob.glob(os.path.join(src, '**', '*.lib'), recursive=True)
            # Also copy x64 contents
            src_x64 = os.path.join(src, 'x64')
            if os.path.exists(src_x64):
                files.extend(glob.glob(os.path.join(src_x64, '*.lib')))

        print(f"\n{folder.upper()}: Found {len(files)} files")
        for f in files:
            try:
                shutil.copy2(f, dest)
                print(f"  Copied: {os.path.basename(f)}")
            except Exception as e:
                print(f"  Error copying {os.path.basename(f)}: {e}")

# Verify
print("\n--- Verification ---")
cudnn_dlls = glob.glob(os.path.join(cuda_bin, 'cudnn*.dll'))
print(f"cuDNN DLLs in CUDA bin: {len(cudnn_dlls)}")
for dll in cudnn_dlls:
    print(f"  {os.path.basename(dll)}")

if len(cudnn_dlls) > 0:
    print("\n✅ cuDNN installation complete!")
    print("You can now run: py filter.py")
else:
    print("\n❌ cuDNN files not found. Please copy manually:")
    print("1. Find the extracted cuDNN folder")
    print("2. Copy all .dll files from bin/ to CUDA 12.4/bin/")
    print("3. Copy all .h files from include/ to CUDA 12.4/include/")
    print("4. Copy all .lib files from lib/x64/ to CUDA 12.4/lib/x64/")

input("\nPress Enter to exit...")
