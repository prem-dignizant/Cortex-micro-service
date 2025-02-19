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

# get_s3_data("https://geometra4-dev.s3.eu-west-1.amazonaws.com/171" , "input_files")
# print(AWS_ACCESS_KEY_ID)


# Image.MAX_IMAGE_PIXELS = None  

# def convert_pdf_to_image(pdf_path, output_folder,page_num):
#     try:
#         page_num = page_num - 1
#         images = convert_from_path(pdf_path, dpi=500)
#         image = images[page_num]

#         img_width, img_height = image.size  # Original image size
#         aspect_ratio = img_width / img_height

#         # Determine new dimensions while maintaining aspect ratio
#         if img_width > img_height:
#             new_width = 1024
#             new_height = int(new_width / aspect_ratio)
#         else:
#             new_height = 1024
#             new_width = int(new_height * aspect_ratio)
        
#         # Resize the image
#         resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         # Create 1024x1024 canvas with white background
#         new_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
#         left = (1024 - new_width) // 2
#         top = (1024 - new_height) // 2
#         new_image.paste(resized_image, (left, top))

#         # Save images
#         os.makedirs(output_folder, exist_ok=True)
#         # high_res_image_path = os.path.join(output_folder, 'high_res_image.jpg')
#         reshaped_image_path = random_file_name(output_folder , "image" , "jpg")
        
#         # image.save(high_res_image_path, 'JPEG')
#         new_image.save(reshaped_image_path, 'JPEG')

#         metadata = {"original_width" : img_width,"original_height" : img_height,"new_width" : new_width,"new_height" : new_height}   

#         return  reshaped_image_path, metadata
#     except Exception as e:
#         return None , None


def convert_pdf_to_images(pdf_path,output_folder,page_num):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        # First image: 1024x1024
        img1 = images[page_num]
        img1 = img1.convert('RGB')
        img1_resized = img1.resize((1024, 1024), Image.Resampling.LANCZOS)
        img1_path = random_file_name(output_folder , "image" , "jpg")
        img1_resized.save(img1_path, "JPEG", quality=95)
        return img1_path      
        # # Second image: 1191x842
        # img2 = images[0]
        # img2 = img2.convert('RGB')
        # img2_resized = img2.resize((1191, 842), Image.Resampling.LANCZOS)
        # img2_path = os.path.join(output_dir, "output_1191x842.jpg")
        # img2_resized.save(img2_path, "JPEG", quality=95)
        # print(f"Created 1191x842 image: {img2_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

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

