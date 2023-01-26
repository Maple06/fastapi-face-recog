# Face Recognition API using FastAPI, OpenCV, and face-recognition library.

### Usage
Build image on main folder, `docker build -t <image-name> .`. Wait for some minute or hour then create a container by running `docker run --name <container-name> -p 8000:8000 <image-name>`.

When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

Send a post request to the main directory "/" (localhost:8000/) that includes 2 body requests, "file" which is an image upload/image binary string and "user_id" which is the user ID to match (string).

This API updates then re-train datasets on 01:00 a.m. local time or when a new user is detected 

### This is a semi-ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage, Waktoo Product.
