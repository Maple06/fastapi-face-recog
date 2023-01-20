print("Importing libraries...")
from fastapi import FastAPI, UploadFile, File, Form
from time import strftime, gmtime
from apscheduler.schedulers.background import BackgroundScheduler
import face_recognition, os, shutil, math, requests, cv2, numpy as np, dlib
print("Libraries imported!")

app = FastAPI()

scheduler = BackgroundScheduler()

status = {'faces': "No Face Detected", "confidence": "0", "match-status": False, "error-status": 1}
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

def update():
    print("Updating datasets...")
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
                    faces = face_cascade.detectMultiScale(gray, 1.4, 7)
                    
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
            pass

    print("Datasets updated!")

todaysUserLen = 0

def encode_faces():
    global face_locations, face_encodings, face_names, known_face_encodings, known_face_names, todaysUserLen

    r = requests.get('https://web.waktoo.com/open-api/get-selfie', headers={'Accept': 'application/json'})
    response = r.json()
    idPerusahaan = 1 # PT Kazee Digital Indonesia
    todaysUserLen = len(response["data"][idPerusahaan-1]["user"])

    update()
    for image in os.listdir('static/faces'):
        face_image = face_recognition.load_image_file(f"static/faces/{image}")
        try:
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(image)
        except IndexError:
            pass

print("Encoding Faces...")
encode_faces()
print("Encoding Done!")

usersLen = 0

def getUsersLen():
    global usersLen, todaysUserLen
    if usersLen != todaysUserLen:
        encode_faces()

scheduler.add_job(getUsersLen, 'interval', seconds=1)
scheduler.start()

update()

print("Running daily dataset update and encoding...")
scheduler.add_job(encode_faces, 'cron', day_of_week='mon-sun', hour=1, minute=00)
scheduler.start()
print("Daily dataset updated and encoded!")

@app.post('/')
def api(user_id : str = Form(...), file: UploadFile = File(...)):
    if user_id == "":
        return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "No user id provided"}

    timeNow = strftime("%d-%b-%y.%H-%M-%S", gmtime())
    filenamesInImages = os.listdir("static/Images")
    filename = f"static/images/api-{timeNow}.png"

    count = 0
    while filename.split("/")[-1] in filenamesInImages:
        filename = f"static/images/api-{timeNow}-{count}.png"
        count += 1

    with open(filename, "wb") as f:
        if file.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
            return {"faceDetected": "No Face Detected", "confidence": "0%", "match-status": False, "error-status": 0, "error-message": "Filename not supported"}
        shutil.copyfileobj(file.file, f)

    IDs_in_dataset = [''.join(i.split(".").pop()) for i in os.listdir("static/faces/")]

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
        small_frame = cv2.resize(small_frame, (0, 0), fx=0.4, fy=0.4)
    else :
        small_frame = frame
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if rgb_small_frame.shape[0] > 600 :
        os.remove(filename)
        return {"faceDetected": status['faces'], "confidence": status["confidence"], "match-status": status["match-status"], "error-status": 0}

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

        detectedFace = status["faces"].split(".")[0]
        if user_id == detectedFace:
            status["match-status"] = True

        face_names.append(f'{name} ({confidence})')

    if status["faces"] == "Unknown" and user_id not in IDs_in_dataset:
        os.remove(filename)
        return {"faceDetected": "Face Detected", "confidence": "N/A", "match-status": False, "error-status": 1, "error-message": "User id not in dataset, but face is detected"}

    # Display the results

    faceDetected = status["faces"]
    confidence = status["confidence"]

    os.remove(filename)
    return {"faceDetected": faceDetected, "confidence": confidence, "match-status": status["match-status"], "error-status": 1}