import boto3
import json
from PIL import Image
from torchvision import transforms

sagemaker_runtime = boto3.client('sagemaker-runtime')
endpoint_name = 'pytorch-inference-2024-05-14-18-01-26-715'

def transform_image(image_bytes):
    # Preprocess input image for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      # Resize images to fixed size
        transforms.ToTensor(),               # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    image = transform(image_bytes).unsqueeze(0)
    return image

# Convert input tensor to JSON format
image = Image.open('petunia.jpeg')
image = transform_image(image)
input_payload = json.dumps({"inputs": image.tolist()})

# Send inference request to the endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=input_payload
)

response_body = response['Body'].read()

# Decode the response body (assuming the response is JSON formatted)
prediction = json.loads(response_body)

print("Prediction:", prediction)