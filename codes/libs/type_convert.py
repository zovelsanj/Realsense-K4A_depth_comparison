import numpy as np
import os

def convert_float64_to_32(data, *, out_dir, idx):
    """Convert float64 data to float32 for faster computation"""
    if isinstance((data[0][0][0]), np.float32):
        return

    data = np.float32(data)
    print(f'Saving data to {out_dir}')
    if idx[0] == 0:
        print(f'Realsense camera data')
        out_dir = os.path.join(out_dir, f'realsense{idx[1]+1}-1-1.npy')
        np.save(out_dir, data)

    elif idx[0] == 1:
        print(f'AzureKinect camera data')
        out_dir = os.path.join(out_dir, f'azurekinect{idx[1]+1}-1-1.npy')
        np.save(out_dir, data)

    else:
        print("Camera type not detected. Please verify camera_id")
        exit(0)
