import cv2 as cv
import os

DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = 5
dataset_size = 200

cap = cv.VideoCapture(0)

for j in range(classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting Data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        cv.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv.LINE_AA)

        cv.imshow("frame", frame)

        if cv.waitKey(25) == ord('q'):


            counter = 0

            while counter<dataset_size:
                ret, frame = cap.read()
                cv.imshow('frame', frame)
                cv.waitKey(25)
                cv.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
                print(counter)
                counter+=1
            print("Complete")
            break

    print(f'Complete for class {j}')

cap.release()
cv.destroyAllWindows()