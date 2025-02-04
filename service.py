import boto3
import requests , os , random
from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime, timedelta

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
REGION_NAME = os.getenv("REGION_NAME")
ZIP_FILE_KEEP = int(os.getenv("ZIP_FILE_KEEP", 1))

def random_file_name(input_folder , prefix , extension):
    while True:
        file_name = f"{prefix}_{random.randint(0, 10000)}.{extension}"
        file_path = os.path.join(input_folder, file_name)
        if not os.path.exists(file_path):  
            return file_path

def get_s3_data(s3_url,input_folder):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,region_name=REGION_NAME)
    file_key = s3_url.split("/")[-1]
    try:
        response = s3_client.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
        file_name = random_file_name(input_folder , "data" , "pdf")

        with open(file_name, 'wb') as file:
            file.write(response['Body'].read())

        return file.name
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return None

# get_s3_data("https://geometra4-dev.s3.eu-west-1.amazonaws.com/1182117" , "input_files")


Image.MAX_IMAGE_PIXELS = None  

def pdf_to_image(pdf_path,output_folder):
    images = convert_from_path(pdf_path, dpi=500)
    image_list = []
    for image in images:

        img_width, img_height = image.size

        # Crop the image by 5% from all sides
        crop_width = int(img_width * 0.05)
        crop_height = int(img_height * 0.05)
        
        # Crop the image (left, upper, right, lower)
        cropped_image = image.crop((
            crop_width, crop_height, 
            img_width - crop_width, img_height - crop_height
        ))

        cropped_width, cropped_height = cropped_image.size
        aspect_ratio = cropped_width / cropped_height

        if cropped_width > cropped_height:
            new_width = 1024
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 1024
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image using the LANCZOS resampling method
        resized_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new blank image with a 1024x1024 white background
        new_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        
        # Calculate the position to paste the resized image onto the white background
        left = (1024 - new_width) // 2
        top = (1024 - new_height) // 2
        new_image.paste(resized_image, (left, top))


        
        # Save the high-resolution image (original image)
        # high_res_image_path = os.path.join(output_folder, 'high_res_image.png')
        # image.save(high_res_image_path, 'PNG')

        reshaped_image_path = random_file_name(output_folder , "image" , "png")

        new_image.save(reshaped_image_path, 'PNG')
        image_list.append(reshaped_image_path)
    os.remove(pdf_path)
    return  image_list


def delete_old_files(output_path):
    """
    Deletes files in the specified directory that are older than ZIP_FILE_KEEP day.
    """
    now = datetime.now()
    one_day_ago = now - timedelta(days=ZIP_FILE_KEEP)
    print(f"Deleting files older than: {one_day_ago}")
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_creation_time < one_day_ago:
                os.remove(file_path)
                print(f"Deleted: {file_path}")



# def pdf_to_image(pdf_path,output_folder):
#     images = convert_from_path(pdf_path)
#     images = convert_from_path(pdf_path, dpi=300)
#     path_list = []
#     for i, image in enumerate(images):
#         image_resized = image.resize((1024, 1024))  
#         image_path = os.path.join(output_folder, f'page_{i + 1}.png')
#         image_resized.save(image_path, 'PNG')
#         path_list.append(image_path)
#     return path_list

