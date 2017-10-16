import os
import tarfile
import zipfile

def _extract(data_dir, filename):
    """
    Extract the tar file
    """
    file_path = os.path.join(data_dir, filename)

    if file_path.endswith(".zip"):
        # Unpack the zip-file.
        zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
    elif file_path.endswith((".tar.gz", ".tgz")):
        # Unpack the tar-ball.
        tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
    print('Done.')

    return
