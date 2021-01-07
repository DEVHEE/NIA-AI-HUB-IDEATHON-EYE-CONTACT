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

import cv2
import numpy as np
import dlib
import math
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray, frame):
    # detect face with CascadeClassifier
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    # landmark
    for (x, y, w, h) in faces:

        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])

        landmarks_display = landmarks[0:68]

        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    return frame

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        dot_top_x = center_top[0]
        dot_top_y = center_top[1]
        dot_bottom_x = center_bottom[0]
        dot_bottom_y = center_bottom[1]

        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 2)

        sleep_length = math.sqrt(math.pow(abs(dot_top_x - dot_bottom_x), 2)) + (math.pow(abs(dot_top_y - dot_bottom_y), 2))

        text = "눈꺼풀 열림 정도: " + str(sleep_length)

        pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        draw.text((50, 50), text, font=ImageFont.truetype('NanumGothic-Bold.ttf', 30), fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)  # 맥

    canvas = detect(gray, frame)

    cv2.imshow("eye", canvas)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()