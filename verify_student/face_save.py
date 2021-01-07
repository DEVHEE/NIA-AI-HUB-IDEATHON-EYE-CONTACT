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
resource_dir = "./resource/for_train/video"
face_dir = "./data/face/image"
face_cnt = 0  # init start frame cnt
face_cntMax = 200  # max frame cnt

# set classifier
face_classifier = cv2.CascadeClassifier("./library/classifier/haarcascade_frontalface_default.xml")

# [START] recognize faces from video resources

# mapping name/id with dir info
resource_dir_nameS_id = glob.glob(resource_dir + "/*")  # list of own face folder

# get name and id
for resource_dir_name_id in resource_dir_nameS_id:
    resource_dir_name_id_Only = os.path.basename(resource_dir_name_id)
    resource_dir_name, resource_dir_nameId_format = resource_dir_name_id_Only.split("_")
    resource_dir_nameId, resource_dir_format = resource_dir_nameId_format.split(".")

    print("Target train video: " + resource_dir_name_id)
    print("Set file name_face: " + resource_dir_name + ", name_id: " + resource_dir_nameId)

    # set own face
    face_name = resource_dir_name
    face_name_id = resource_dir_nameId

    resource_pathVid = resource_dir_name_id
    vid = cv2.VideoCapture(resource_pathVid)

    face_cnt = 0

    imgCnt = 0


    while vid.isOpened():
        ret, frame = vid.read()

        if ret:
            img = frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # image to grayscale
            face = face_classifier.detectMultiScale(gray, 1.25, 10)  # recognize faces algorithm

            for (x, y, w, h) in face:
                # set ROI of faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(face_cnt), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                face_cnt += 1

                imgCnt = imgCnt + len(resource_dir_nameS_id)

                # save ROI of faces
                ROI_face = gray[y:y + h, x:x + w]
                ROI_face = cv2.resize(ROI_face, (250, 250))
                face_dir_name = str(face_dir + "/" + face_name + "_" + face_name_id)
                if not os.path.exists(face_dir_name):
                    os.makedirs(face_dir_name)
                face_pathImg = os.path.join(face_dir_name, str(face_cnt) + ".jpg")
                cv2.imwrite(face_pathImg, ROI_face)

                print("Detected Face(" + face_name + "_" + face_name_id + "): " + str(face_cnt) + ", Saved Path: " + face_pathImg)

        # control window
        cv2.imshow("Now saving faces", frame)
        if cv2.waitKey(1) > -1 or face_cnt == face_cntMax:
            break

    print("")
# [END] recognize faces from video resources

cv2.destroyAllWindows()
cv2.waitKey(1)

print("Saving " + str(imgCnt) + " picture(s) / " + str(len(resource_dir_nameS_id)) + " model(s) completed. " + str(resource_dir_nameS_id))
