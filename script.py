import numpy as np
import pyarrow.parquet as pq
import joblib
import argparse

class ImageDatasetFromParquet(torch.utils.data.Dataset):
    def __init__(self, filename):
        super().__init__()
        self.file = pq.ParquetFile(filename)

    def __getitem__(self, idx):
        data = self.file.read_row_group(idx).to_pydict()
        X_jet = np.array(data["X_jet"][0], dtype=np.float32)[0:4, :, :]
        genM = np.array(data["m"][0], dtype=np.float32)
        iphi = np.array(data["iphi"][0], dtype=np.float32) / 360.0
        ieta = np.array(data["ieta"][0], dtype=np.float32) / 170.0

        return {
            "X_jets": X_jet.transpose(2, 1, 0),
            "m": genM,
            "pt": np.array(data["pt"][0], dtype=np.float32),
            "ieta": ieta,
            "iphi": iphi,
        }

    def __len__(self):
        return self.file.num_row_groups

def process_chunks(start_chunk, end_chunk):
    dataset = ImageDatasetFromParquet(filename='top_gun_opendata_6.parquet')
    total_samples = len(dataset)
    chunk_size = total_samples // 5  # Process in 20% chunks

    data_dict = {
        "X_jets": [],
        "m": [],
        "pt": [],
        "ieta": [],
        "iphi": []
    }

    for chunk_idx in range(start_chunk, end_chunk):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_samples)
        print(f"Processing samples {chunk_start} to {chunk_end - 1}...")

        chunk_X_jets = np.zeros((chunk_end - chunk_start, 125, 125, 8), dtype=np.float32)
        chunk_m = np.zeros(chunk_end - chunk_start, dtype=np.float32)
        chunk_pt = np.zeros(chunk_end - chunk_start, dtype=np.float32)
        chunk_ieta = np.zeros(chunk_end - chunk_start, dtype=np.float32)
        chunk_iphi = np.zeros(chunk_end - chunk_start, dtype=np.float32)

        for idx in range(chunk_start, chunk_end):
            sample = dataset[idx]
            chunk_X_jets[idx - chunk_start] = sample["X_jets"]
            chunk_m[idx - chunk_start] = sample["m"]
            chunk_pt[idx - chunk_start] = sample["pt"]
            chunk_ieta[idx - chunk_start] = sample["ieta"]
            chunk_iphi[idx - chunk_start] = sample["iphi"]

        data_dict["X_jets"].append(chunk_X_jets)
        data_dict["m"].append(chunk_m)
        data_dict["pt"].append(chunk_pt)
        data_dict["ieta"].append(chunk_ieta)
        data_dict["iphi"].append(chunk_iphi)

        # Save intermediate results after each chunk
        chunk_filename = f"chunk_{chunk_idx}.joblib"
        joblib.dump(data_dict, chunk_filename)
        print(f"Intermediate results saved to {chunk_filename}")

    return data_dict

def combine_chunks(num_chunks):
    combined_data_dict = {
        "X_jets": [],
        "m": [],
        "pt": [],
        "ieta": [],
        "iphi": []
    }

    for chunk_idx in range(num_chunks):
        chunk_filename = f"chunk_{chunk_idx}.joblib"
        data_dict = joblib.load(chunk_filename)
        combined_data_dict["X_jets"].extend(data_dict["X_jets"])
        combined_data_dict["m"].extend(data_dict["m"])
        combined_data_dict["pt"].extend(data_dict["pt"])
        combined_data_dict["ieta"].extend(data_dict["ieta"])
        combined_data_dict["iphi"].extend(data_dict["iphi"])

    combined_data_dict["X_jets"] = np.concatenate(combined_data_dict["X_jets"], axis=0)
    combined_data_dict["m"] = np.concatenate(combined_data_dict["m"], axis=0)
    combined_data_dict["pt"] = np.concatenate(combined_data_dict["pt"], axis=0)
    combined_data_dict["ieta"] = np.concatenate(combined_data_dict["ieta"], axis=0)
    combined_data_dict["iphi"] = np.concatenate(combined_data_dict["iphi"], axis=0)

    np.savez('data_06.npz', **combined_data_dict)
    print("Data saved as a single .npz file.")

    import zipfile
    import os

    zip_filename = 'data_06.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write('data_06.npz')

    print("Zip file created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and combine data chunks')
    parser.add_argument('--start_chunk', type=int, default=0, help='Start chunk index')
    parser.add_argument('--end_chunk', type=int, default=5, help='End chunk index')
    parser.add_argument('--combine', action='store_true', help='Combine existing chunks')
    args = parser.parse_args()

    if args.combine:
        combine_chunks(args.end_chunk)
    else:
        process_chunks(args.start_chunk, args.end_chunk)
