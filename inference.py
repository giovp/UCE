# This file is used to define the necessary functions to deploy the SCVI model to SageMaker
# model_fn, input_fn, predict_fn, output_fn are required to be implemented by SageMaker
# More info on the dir structure: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#model-directory-structure
# These get packaged up into a .tar.gz file, uploaded to an S3 bucket, and then deployed to SageMaker via deploy.py

import json
from anndata import read_h5ad
from omegaconf import OmegaConf
import yaml
from evaluate import AnndataProcessor
from utils import download_s3_directory, download_s3_file
import os
import logging
import os
import numpy as np
import argparse
from accelerate import Accelerator
from io import BytesIO

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """Download the model files from S3
    Since the real model files are are over 25 GB, model_fn will download the model files from S3. model_dir is simply a placeholder.
    The model files should be stored in the /tmp/model_files directory.
    """
    local_model_dir = "/tmp/model_files"  # SageMaker provides /tmp for temporary storage
    uce_model_files_s3_path = "s3://generate-cross-species/models/uce_assets_03_12_25/"
    download_s3_directory(uce_model_files_s3_path, local_model_dir)
    return local_model_dir


def input_fn(request_body, request_content_type):
    """
    Deserialize and preprocess the incoming request for prediction.

    This function is called by SageMaker to parse and process each incoming inference request.
    It handles both synchronous and asynchronous requests by accepting a request body and its
    content type. The function expects the request body to be in JSON format with specific
    keys indicating the S3 location of the input data and the organism type.

    Args:
        request_body (bytes): The raw request payload sent by the client.
        request_content_type (str): The MIME type of the incoming request.

    Returns:
        dict: A dictionary containing the preprocessed `AnnData` object and the `organism` identifier.

    Raises:
        ValueError: If the `request_content_type` is not supported or required keys are missing in the JSON.
        json.JSONDecodeError: If the `request_body` is not valid JSON.
    """
    logger.info(f"Input function called with content type: {request_content_type}")
    logger.info(f"Request body: {request_body}")
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # Parse the input JSON
    input_dict = json.loads(request_body)
    logger.info(f"Input data: {input_dict}")

    # Get the adata_path from the input_dict and download the file from S3
    adata_path = input_dict["adata_path"]
    download_s3_file(adata_path, "/tmp/")

    args = argparse.Namespace(**input_dict)

    return args


def predict_fn(input_data, model):
    logger.info("Starting prediction function.")
    logger.info(f"Input data: {input_data}")
    logger.info(f"Model: {model}")

    accelerator = Accelerator(project_dir=".")
    logger.info("Accelerator initialized.")

    processor = AnndataProcessor(input_data, accelerator)
    logger.info("AnndataProcessor initialized.")

    processor.preprocess_anndata()
    logger.info("Anndata preprocessing completed.")

    processor.generate_idxs()
    logger.info("Index generation completed.")

    embedding_adata = processor.run_evaluation()
    logger.info("Evaluation completed.")

    return embedding_adata.obsm["X_uce"]


def output_fn(prediction, content_type):
    """
    Serialize the prediction output into the desired response format.

    This function converts the prediction result into a format suitable for the client,
    based on the specified `content_type`. It supports both binary NumPy arrays and JSON
    serialization, allowing clients to reconstruct the prediction accurately.

    Args:
        prediction (Any): The prediction result to be serialized. Typically, this is one or more
                          NumPy `ndarray` objects.
        content_type (str): The desired MIME type for the response. Supported types are
                            `'application/x-npy'` for NumPy binary format and `'application/json'`.

    Returns:
        tuple: A tuple containing the serialized prediction and the corresponding content type.

    Raises:
        ValueError: If the specified `content_type` is not supported.
    """
    logger.info(f"Output function called with content type: {content_type}")
    logger.info(f"Prediction type: {type(prediction)}")
    logger.info(f"Prediction shape: {prediction.shape}")

    if content_type == "application/x-npy":
        # Convert NumPy array to binary stream in NumPy .npy format
        buffer = BytesIO()
        np.save(buffer, prediction)
        return buffer.getvalue(), "application/x-npy"
        # When making a request to the endpoint, the response is a binary stream in NumPy .npy format so we can reconstruct the array using np.load(io.BytesIO(response_body)

    elif content_type == "application/json":
        # Convert NumPy array to JSON
        response_body = prediction.tolist()
        return json.dumps(response_body), "application/json"

    else:
        # Default to NumPy binary format if content_type is unsupported
        buffer = BytesIO()
        np.save(buffer, prediction)
        return buffer.getvalue(), "application/x-npy"
