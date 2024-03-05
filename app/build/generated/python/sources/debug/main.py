import cv2
import mediapipe as mp
import math
import winsound
import win32com.client
import threading
from ultralytics import YOLO
from threading import Thread

speak_out = win32com.client.Dispatch("SAPI.SPVOICE")

model = YOLO("best.pt")

label_mapping = {
    0: "眼和触角",
    1: "头",
    2: "壳",
    3: "身体"
}


class ThreadWithReturnValue(Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]

    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 65535.

    if angle_ > 180.:
        angle_ = 65535.

    return angle_


def hand_angle(hand_):
    angle_list = []

    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)

    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)

    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)

    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)

    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)

    return angle_list


def h_gesture(angle_list):
    thr_angle = 65.
    thr_angle_s = 49.
    gesture_str = None

    if 65535. not in angle_list:
        if (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "one"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
            gesture_str = "five"
        else:
            gesture_str = "other"

    return gesture_str


def get_detection_cls(yolo_results):
    detection_cls_results = []
    for i in yolo_results:
        k = i.boxes.xyxy.tolist()[0]
        cls = int(i.boxes.cls.tolist()[0])
        x1, y1, x2, y2 = k[0], k[1], k[2], k[3]
        detection_cls_results.append([[x1, y1, x2, y2, cls]])
    return detection_cls_results


def speak(str):
    speak_out.Speak(str)
    winsound.PlaySound(str, winsound.SND_ASYNC)


def blue_dot_thread(cx, cy, get_results, label_mapping):
    for detection in get_results:
        if cx > detection[0][0] and cx < detection[0][2] and cy > detection[0][1] and cy < detection[0][3]:
            speak("这是蜗牛的" + label_mapping[detection[0][4]])


def detect():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)
    cap = cv2.VideoCapture(0)

    previous_gesture = None

    while True:

        ret, frame = cap.read()
        yolo_results = model.predict(frame)
        thread = ThreadWithReturnValue(target=get_detection_cls, args=(yolo_results))
        thread.start()
        get_results = thread.join()

        annotated_frame = yolo_results[0].plot()
        frame[0:annotated_frame.shape[0], 0:annotated_frame.shape[1]] = annotated_frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 8:
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (225, 0, 0), cv2.FILLED)

                hand_local = []

                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))

                if hand_local:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    if gesture_str == "one" and previous_gesture != gesture_str and get_results:
                        thread = threading.Thread(target=blue_dot_thread, args=(cx, cy, get_results, label_mapping))
                        thread.start()
                    previous_gesture = gesture_str

        cv2.imshow('MangTu', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()


if __name__ == '__main__':
    detect()