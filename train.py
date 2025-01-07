import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearnModel
from sagemaker.s3 import S3Downloader

# Set up SageMaker environment
role = get_execution_role()
session = sagemaker.Session()

# Upload dataset and scripts to S3
bucket_name = session.default_bucket()
prefix = 'ml-model'

# Assuming dataset and scripts are already available in the local directory
train_data = session.upload_data(path='train.csv', bucket=bucket_name, key_prefix=prefix)

# Define the SKLearn estimator for SageMaker
estimator = sagemaker.sklearn.estimator.SKLearnModel(
    entry_point='train.py',
    role=role,
    framework_version='0.23-1',
    instance_type='ml.m5.large',
    instance_count=1,
    output_path=f's3://{bucket_name}/{prefix}/output'
)

# Start training job
estimator.fit({'train': train_data})

