import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api
# load_dotenv()  # Load .env variables into os.environ
print("API key loaded:", os.getenv("CLOUDINARY_API_KEY") is not None)


load_dotenv()
import os
from dotenv import load_dotenv
load_dotenv()

import cloudinary
import cloudinary.uploader
import cloudinary.api

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image(path, folder="urban_threads"):
    res = cloudinary.uploader.upload(path, folder=folder)
    return res["secure_url"]
