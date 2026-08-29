import urllib.request
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vmtable_vmtable.csv.gz"
    filename = "trace_data_vmtable_vmtable.csv.gz"
    output_path = data_dir / filename

    print(f"Target directory: {data_dir}")
    print(f"Downloading {url}...")

    def reporthook(blocknum, blocksize, totalsize):
        readed = blocknum * blocksize
        if totalsize > 0:
            percent = min(int(readed * 100 / totalsize), 100)
            print(f"\rDownloading: {percent}% ({readed}/{totalsize} bytes)", end="", flush=True)
        else:
            print(f"\rDownloaded {readed} bytes", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print(f"\nSuccessfully downloaded to {output_path}")


if __name__ == "__main__":
    main()
