# Face Recognition Website using Flask, OpenCV, and face-recognition library.
### Features
1. Live webcam
2. Picture taking
3. Face recognition from recent taken picture
4. Face recognition API from internet pictures

### How To Use
Get into localhost:5000 after running app.py, on landing page ("/" directory) webcam will be shown with a take picture button. When button pressed picture will be saved in local folder and image will be processed and shown at /result directory.
REST API is on /api directory with a mandatory "l" argument. (localhost:5000/api?l=https://example.com/test.jpg) l argument must be an internet image link.

### This is a intern test module for PT Kazee Digital Indonesia, Waktoo Product.
