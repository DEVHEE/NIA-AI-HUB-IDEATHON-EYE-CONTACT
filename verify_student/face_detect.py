"""
NIA-AI-HUB-IDEATHON-EYE-CONTACT

AI 허브 인공지능 학습용 데이터 활용 아이디어 공모전 - 실현 가능 서비스 부문

<팀 토니스빠끄>

[아이컨택(안면인식 및 동작인식을 활용한 온라인 수업 감독 및 보조 서비스)]

팀장: 김동희 / 도쿄 고쿠시칸대학 20 이공학부 이공학과 전자정보컴퓨터공학
팀원1: 강재원 / 상지대학교 20 융합기술공과대학 정보통신소프트웨어공학
팀원2: 이서진 / 건국대학교 20 사범대학 교육공학

LEADER: KIM DONGHEE / TOKYO KOKUSHIKAN UNIVERSITY 20, School of Science and Engineering, Faculty of Science and Technology, Department of Electronics and Informatics
MEMBER1: KANG JAEWON / SANGJI UNIVERSITY 20, College of Science & Engineering, Department of Information and Communication Engineering
MEMBER2: LEE SEOJIN / KONKUK UNIVERSITY 20, College of Education, Department of Educational Technology

COPYRIGHT © 2021 KIM DONGHEE / KANG JAEWON / LEE SEOJIN. ALL RIGHTS RESERVED.
We sincerely hope for the end of COVID-19 in Republic of Korea and all over the world.
"""

# import modules
import cv2
import os
import glob

# set var
face_dir = "./data/face/image"
train_dir = "./data/face/trained"

# set classifier
face_classifier = cv2.CascadeClassifier("./library/classifier/haarcascade_frontalface_default.xml")

# create LBP and read trained xml
modelTrain = cv2.face.LBPHFaceRecognizer_create()
modelTrain.read(os.path.join(train_dir, "faces.xml"))

# mapping name/id with dir info
face_dir_nameS_id = glob.glob(face_dir + "/*")  # list of own face folder
face_nameS = dict([])
for face_dir_name_id in face_dir_nameS_id:
    face_dir_name_id = os.path.basename(face_dir_name_id)
    face_name, face_dir_nameId = face_dir_name_id.split("_")
    face_nameS[int(face_dir_nameId)] = face_name

# read video
resource_pathVid = "./resource/for_detect/video/detect_thequiett_0.mp4"
vid = cv2.VideoCapture(resource_pathVid)

# [START] detect with for_train vid
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        print("Error: Wrong video file.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    face = face_classifier.detectMultiScale(gray, 1.25, 10)  # recognize faces algorithm

    for (x, y, w, h) in face:
        # set ROI of faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ROI_face = frame[y:y + h, x:x + w]
        ROI_face = cv2.resize(ROI_face, (250, 250))  # match with face_save img size
        ROI_face = cv2.cvtColor(ROI_face, cv2.COLOR_BGR2GRAY)

        # predict with LBP Face Detection
        faceId, faceConf = modelTrain.predict(ROI_face)  # set confidence
        if faceConf < 400:
            faceAccu = int((1 - faceConf / 400) * 100)
            if faceAccu > 80:
                faceLabel = face_nameS[faceId] + " / " + str(faceAccu) + "%"
                # faceLabel = (face_nameS[faceId], faceAccu)
            else:
                faceLabel = "(None)" + " / " + str(faceAccu) + "%"

            # print faceLabel on face rect
            cv2.putText(frame, faceLabel, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print("Detected Face / Accuracy : " + faceLabel)
# [END] detect with for_train vid

    cv2.imshow("Now detecting faces", frame)
    cv2.waitKey(1)
