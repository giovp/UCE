"""
Utils

"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
import tarfile
import boto3
from urllib.parse import urlparse
import anndata as ad

def get_shapes_dict(dataset_path):
    shapes_dict = {}
    datasets_df = pd.read_csv(dataset_path)
    sorted_dataset_names = sorted(datasets_df["names"])

    for name in sorted_dataset_names:
        shapes_dict[name] = (int(datasets_df.set_index("names").loc[name]["num_cells"]), 8000)

    shapes_dict["dev_immune_mouse"] = (443697, 4786)
    shapes_dict["dev_immune_human"] = (34009, 5566)
    shapes_dict["intestinal_tract_human"] =  (69668, 5192)
    shapes_dict["gtex_human"] =  (18511, 7109)
    shapes_dict["gut_endoderm_mouse"] =  (113043, 6806)
    shapes_dict["luca"] =  (249591, 7196)
    shapes_dict.update({
     "madissoon_novel_lung":(190728, 8000),
     'flores_cerebellum_human': (20232, 8000),
     'osuch_gut_human': (272310, 8000),
     'msk_ovarian_human': (929690, 8000),
     'htan_vmuc_dis_epi_human': (65084, 8000),
     'htan_vmuc_val_epi_human': (57564, 8000),
     'htan_vmuc_non_epi_human': (9099, 8000),
     'hao_pbmc_3p_human': (161764, 8000),
     'hao_pbmc_5p_human': (49147, 8000),
     'gao_tumors_human': (36111, 8000),
     'swabrick_breast_human': (92427, 8000),
     'wu_cryo_tumors_human': (105662, 8000),
     'cell_line_het_human': (53513, 8000),
     'bi_allen_metastasis_human': (27787, 8000),
     'zheng68k_human': (68579, 8000),
     'zheng68k_12k_human': (68579, 12000),
     'mouse_embryo_ct': (153597, 12000),
     "regev_gtex_heart": (36574, 8000),
     "tabula_sapiens_heart": (11505, 8000),
     "10k_pbmcs":(11990, 12000),
     "epo_ido":(35834,12000),
     'tabula_sapiens_kidney': (9641, 8000),
     'tabula_microcebus_kidney': (14592, 8000),
     'tabula_muris_kidney': (2781, 8000),
     'tabula_muris_senis_kidney': (19610, 8000),
      'immune_human': (33506, 8000)
                       })

    shapes_dict["zyl_sanes_glaucoma_pig"] = (5901, 6819)
    shapes_dict["parkinsons_macaF"] = (1062, 5103)

    for row in datasets_df.iterrows():
        ngenes = row[1].num_genes
        ncells = row[1].num_cells
        name = row[1].names
        if not np.isnan(ngenes):
            shapes_dict[name] = (int(ncells), int(ngenes))

    return shapes_dict


def figshare_download(url, save_path):
    """
    Figshare download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """

    if os.path.exists(save_path):
        return
    else:
        # Check if directory exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        print("Downloading " + save_path + " from " + url + " ..." + "\n")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB',
                            unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    # If the downloaded filename ends in tar.gz then extraact it
    if save_path.endswith(".tar.gz"):
       with tarfile.open(save_path) as tar:
            tar.extractall(path=os.path.dirname(save_path))
            print("Done!")


def is_s3_url(path):
    """Return True if the provided path is an S3 URL."""
    return isinstance(path, str) and path.startswith("s3://")

def download_s3_file(s3_url, dest_folder):
    """
    Download a file from an S3 URL to the specified destination folder.
    
    Parameters:
    - s3_url (str): The S3 URL (e.g. s3://bucket/key).
    - dest_folder (str): The local folder where the file should be saved.
    
    Returns:
    - local_filename (str): The path to the downloaded file.
    """
    os.makedirs(dest_folder, exist_ok=True)
    # Remove the "s3://" prefix and split the rest into bucket and key
    without_prefix = s3_url.replace("s3://", "", 1)
    bucket, key = without_prefix.split("/", 1)
    local_filename = os.path.join(dest_folder, os.path.basename(key))
    boto3.client("s3").download_file(bucket, key, local_filename)
    return local_filename

def download_s3_directory(s3_url, dest_folder):
    """
    Recursively download all files from an S3 URL representing a directory.
    
    Parameters:
    - s3_url (str): The S3 URL for the directory (e.g. s3://bucket/path/to/dir).
    - dest_folder (str): The local folder where the directory contents should be saved.
    
    Returns:
    - dest_folder (str): The path to the directory where the files were downloaded.
    """
    os.makedirs(dest_folder, exist_ok=True)
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in result.get('Contents', []):
            s3_object_key = obj['Key']
            # Compute the local path relative to the prefix
            relative_path = os.path.relpath(s3_object_key, prefix)
            local_file = os.path.join(dest_folder, relative_path)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            client.download_file(bucket, s3_object_key, local_file)
    return dest_folder

def process_adata(adata_path):
    """
    Process an AnnData object from a file path.
    
    Args:
        adata_path (str): The path to the AnnData object.

    Returns:
        AnnData: The processed AnnData object.
    """
    
    adata = ad.read_h5ad(adata_path)
    # set features to be gene symbols which is required
    # by evaluate.AnndataProcessor
    adata.var_names = pd.Index(list(adata.var["feature_name"]))
    adata.var_names = adata.var["feature_name"].values
    return adata


def handle_s3_download(args, attr_name, dest, download_func, post_func=None):
    """
    Checks if the attribute value is an S3 URL and downloads it if needed.

    Parameters:
    -----------
    args : argparse.Namespace
        The parsed command-line arguments.
    attr_name : str
        The name of the attribute in args.
    dest : str or Path
        The destination directory/path to download the file/directory.
    download_func : function
        The function to use for downloading (e.g., download_s3_file or download_s3_directory).
    post_func : function, optional
        An optional post-processing function that further processes the downloaded file.
    """
    value = getattr(args, attr_name)
    if value and is_s3_url(value):
        new_value = download_func(value, dest)
        if post_func:
            new_value = post_func(new_value)
        setattr(args, attr_name, new_value)

def download_model_from_s3(s3_bucket, s3_model_prefix, local_dir):
    """Download model files from S3 to the SageMaker container's local storage."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_model_prefix)

    os.makedirs(local_dir, exist_ok=True)
    
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                local_file_path = os.path.join(local_dir, os.path.basename(s3_key))
                s3.download_file(s3_bucket, s3_key, local_file_path)
                print(f"Downloaded {s3_key} to {local_file_path}")
