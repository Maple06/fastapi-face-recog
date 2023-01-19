from fastapi import FastAPI
from time import strftime, gmtime
import face_recognition, os, shutil, math, requests, cv2, numpy as np, dlib


app = FastAPI()

status = {'faces': "No Face Detected", "confidence": "0", "match-status": False, "error-status": 1}

takePhotoReq = False

detector = dlib.get_frontal_face_detector()

recentPicTaken = ""

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []

def encode_faces():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names
    for image in os.listdir('static/faces'):
        face_image = face_recognition.load_image_file(f"static/faces/{image}")
        try:
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(image)
        except IndexError:
            pass

encode_faces()

@app.get('/')
def api(l: str = ""):
    picLink = l

    if picLink == None or picLink == "":
        return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "No link argument found"}
    response = requests.get(picLink, stream=True)
    timeNow = strftime("%d-%b-%y.%H-%M-%S", gmtime())
    filename = f"static/images/api-{timeNow}.png"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

    frame = cv2.imread(f"static/images/api-{timeNow}.png")

    try:
        if filename == None:
            return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "Not a valid filename"}
    except ValueError:
        pass

    global status

    status["faces"] = "No Face Detected"
    status["confidence"] = "0%"
    status["match-status"] = False
    status["error-status"] = 1

    # Upscale image
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'FSRCNN_x4.pb'
    sr.readModel(path)
    sr.setModel("fsrcnn", 4)
    
    if frame.shape[0] >= 1000 or frame.shape[1] >= 1000:
        small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    elif frame.shape[0] <= 400 or frame.shape[1] <= 400 :
        small_frame = sr.upsample(frame)
        small_frame = cv2.resize(small_frame, (0, 0), fx=0.5, fy=0.5)
    else :
        small_frame = frame
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if rgb_small_frame.shape[0] > 600 :
        return {"faceDetected": faceDetected, "confidence": confidence, "match-status": status["match-status"], "error-status": 0}

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = '???'

        # Calculate the shortest distance to face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])

        status["faces"] = name
        status["confidence"] = confidence

        if picLink[:58] == "https://waktoo-selfie.obs.ap-southeast-3.myhuaweicloud.com" or picLink[:60] == "https:\/\/waktoo-selfie.obs.ap-southeast-3.myhuaweicloud.com":
            checkedID = picLink.replace("\\", "").split("/")[-1].split("_")[0]
            detectedFace = status["faces"].split(".")[0]
            if checkedID == detectedFace:
                status["match-status"] = True
            else:
                status["match-status"] = False

        face_names.append(f'{name} ({confidence})')

    # Display the results

    faceDetected = status["faces"]
    confidence = status["confidence"]

    return {"faceDetected": faceDetected, "confidence": confidence, "match-status": status["match-status"], "error-status": 1}

@app.get("/update")
def update():
    r = requests.get('https://web.waktoo.com/open-api/get-selfie', headers={'Accept': 'application/json'})

    response = r.json()
    idPerusahaan = 1 # PT Kazee Digital Indonesia
    response = response["data"][idPerusahaan-1]["user"]

    for i in response:
        count = 1
        try:
            for j in i["foto"]:
                url = j["foto_absen"]

                r = requests.get(url)

                filename = f'static/faces/{i["user_id"]}.png'

                with open(filename, 'wb') as f:
                    f.write(r.content)         
                try :
                    img = cv2.imread(filename)
                    # Convert into grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Load the cascade
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.4, 6)
                    
                    # Draw rectangle around the faces and crop the faces
                    for (x, y, w, h) in faces:
                        faces = img[y:y + h, x:x + w]
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    path = 'FSRCNN_x4.pb'
                    sr.readModel(path)
                    sr.setModel("fsrcnn", 4)
                    upscaled = sr.upsample(faces)
                    cv2.imwrite(filename, upscaled)
                    break
                except :
                    pass  
                os.remove(filename)         
        except IndexError:
            print("jumlah foto: 0")
    
    return {"status": "success"}