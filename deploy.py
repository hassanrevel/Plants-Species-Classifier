import os

from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchModel
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Specify the S3 path where your model artifacts are stored
model_data = 's3://discriminative-ai-30/PlantSpeciesClassifier/models/model.tar.gz'
role = os.environ['SAGEMAKER_EXECUTION_ROLE']
image_uri = '763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.2.0-cpu-py310-ubuntu20.04-sagemaker-v1.0'

# Create a PyTorchModel object
model = PyTorchModel(
    entry_point="inference.py",
    # source_dir="code",
    role=role,
    model_data=model_data,
    image_uri=image_uri
)

model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)