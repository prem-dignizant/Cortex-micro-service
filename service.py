import boto3
import requests , os , random
from pdf2image import convert_from_path

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
region_name = os.getenv("region_name")

def get_s3_data(s3_url):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,region_name=region_name)
    file_key = s3_url.split("/")[-1]

    try:
        # Download from S3
        response = s3_client.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
        with open(f"input_files/data_{random.randint(0, 10000)}.pdf", 'wb') as file:
            file.write(response['Body'].read())
        # import pdb; pdb.set_trace()
        return file.name
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return None

# get_s3_data("s3://prem272buck/Mahesh Maniya_CV.pdf")


def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    images = convert_from_path(pdf_path, dpi=300)
    path_list = []
    for i, image in enumerate(images):
        image_resized = image.resize((1024, 1024))  
        image_path = os.path.join('input_files', f'page_{i + 1}.png')
        image_resized.save(image_path, 'PNG')
        path_list.append(image_path)
    return path_list