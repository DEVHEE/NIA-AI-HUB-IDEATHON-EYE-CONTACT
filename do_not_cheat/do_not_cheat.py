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

import numpy as np
import dlib
import cv2
from do_not_cheat.gaze_tracking import GazeTracking
from PIL import ImageFont, ImageDraw, Image

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


def detect(gray, frame):
    # detect face with CascadeClassifier
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    # Landmark
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        landmarks_display = landmarks[36:48]

        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    return frame


while True:
    _, frame = webcam.read()

    gaze.refresh(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "깜빡임\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    elif gaze.is_right():
        text = "오른쪽 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    elif gaze.is_left():
        text = "왼쪽 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    elif gaze.is_center():
        text = "정면 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
    else:
        text = "눈동자 인식 불가"

    pill_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pill_image)
    draw.text((50, 50), text, font=ImageFont.truetype('NanumGothic-Bold.ttf', 30), fill=(0, 255, 0))
    frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    cv2.putText(frame, str(left_pupil), (280, 135), cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 2)
    cv2.putText(frame, str(right_pupil), (305, 195), cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 2)
    frame = detect(gray, frame)
    cv2.imshow("Sleep", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
