import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

labels_dict = {'0': 'YES', '1': 'NO', '2': 'HELLO', '3': 'THANK YOU', '4': 'PLEASE'}


cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

while True:
    data_aux = []
    x_ = []
    y_ = []
    offset = 20
    ret, img = cap.read()

    H, W, _  = img.shape

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in result.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int((min(x_)*W) - offset)
        x2 = int(offset + (max(x_)*W))
        y1 = int((min(y_)*H) - offset)
        y2 = int(offset + (max(y_)*H))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = labels_dict[str(int(prediction[0]))]

        cv.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 4)
        cv.putText(img, predicted_char, (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                   cv.LINE_AA)

    cv.imshow('frame', img)
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()