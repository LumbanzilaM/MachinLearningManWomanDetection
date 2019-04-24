import subprocess
import os
from tensorflow.python.lib.io import file_io

WORKING_DIR = os.getcwd()

def load_data(path, destination):
    """Verifies if file is in Google Cloud.

    Args:
    path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
    destination: (str) The filename to save as on local disk.

    Returns:
    A filename
    """
    if path.startswith('gs://'):
        ret = download_files_from_gcs(path, destination=destination)
        return ret
    return path


def download_files_from_gcs(source, destination):
    # Download files from GCS to a WORKING_DIR/.
    # Copy raw files from GCS into local path.

    raw_local_files_data_path = os.path.join(WORKING_DIR, destination)
    subprocess.check_output(['gsutil', '-m', 'cp', source, raw_local_files_data_path], shell=True)
    return raw_local_files_data_path


def upload_file_to_gcs(source, destination):

    # Copy raw files from GCS into local path.
    subprocess.check_call(['gsutil', '-m', 'cp', source, destination], shell=True)

def upload_dir_to_gcs(source, destination):
    # Copy raw files from GCS into local path.
    #subprocess.check_call(['gsutil', '-m', 'cp', '-r', source, destination], shell=True)
    subprocess.check_call(
        ['gsutil', '-m', 'cp', '-r', "logs", "gs://scancube-ml-storage/Datasets/5labels/Models/tmp"],
        shell=True)


def save_file_in_cloud(source, destination):
    with file_io.FileIO(source, 'r') as infile:
        with file_io.FileIO(destination, mode='w+') as outfile:
            outfile.write(infile.read())
