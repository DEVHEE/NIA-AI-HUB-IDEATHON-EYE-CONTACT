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
import numpy as np
import os
import glob

# set ver
face_dir = "./data/face/image"
train_dir = "./data/face/trained"
trainData = []  # set train data array
trainIdS = []  # set train id array
imgCnt = 0

face_dir_nameS_id = glob.glob(face_dir + "/*")  # list of own face folder

print("Targeting train model data files...")

# get id of name
for face_dir_name_id in face_dir_nameS_id:
    face_dir_nameId = face_dir_name_id.split("_")[1]

    face_dir_name_id_imageS = glob.glob(face_dir_name_id + "/*.jpg")  # list of own image of face ids
    print("Target: %s, %d file(s)" % (face_dir_name_id, len(face_dir_name_id_imageS)))

    for face_dir_name_id_image in face_dir_name_id_imageS:
        trainImage = cv2.imread(face_dir_name_id_image, cv2.IMREAD_GRAYSCALE)  # read train image

        # create train data/ids array
        trainData.append(np.asarray(trainImage, dtype=np.uint8))
        trainIdS.append(int(face_dir_nameId))

    imgCnt = imgCnt + len(face_dir_name_id_imageS)

# convert to np array
trainData = np.asarray(trainData)
trainIdS = np.int32(trainIdS)

# create LBP and train
print("Creating LBPH Face Recognizer...")
modelTrain = cv2.face.LBPHFaceRecognizer_create()
print("Training model(s)...")
modelTrain.train(trainData, trainIdS)

# make xml data file
if not os.path.exists(train_dir):
    print("Warning: There is not a directory for the trained data file. Creating directory...")
    os.makedirs(train_dir)
modelTrain.write(train_dir + "/faces.xml")
print("Training total " + str(len(face_dir_nameS_id)) + " model(s), " + str(imgCnt) + " image(s) completed.")
