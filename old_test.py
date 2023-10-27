import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('SaveTube.App-CCTV Video of a parking lot.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

areas = {
    1: [(794, 133), (936, 145), (988, 205), (822, 190)],
    2: [(630, 118), (793, 129), (811, 185), (619, 179)],
    3: [(560, 241), (849, 245), (868, 440), (500, 403)],
    4: [(335, 90), (435, 104), (390, 141), (279, 141)],
    5: [(466, 96), (572, 104), (527, 149), (430, 149)],
    6: [(202, 89), (284, 94), (215, 132), (117, 130)],
    7: [(52, 198), (200, 212), (43, 361), (2, 234)],
    8: [(845, 259), (1019, 262), (1016, 433), (872, 429)],
    9: [(82, 91), (124, 96), (72, 139), (8, 139)],
    10: [(334, 225), (564, 241), (469, 420), (142, 372)],
    11: [(8, 55), (1019, 55), (1015, 445), (6, 448)],
}

total_car_count = 0

counted_cars = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    area_objects = {area_num: [] for area_num in areas}
    object_counts = {area_num: 0 for area_num in areas}

    for obj in px.to_numpy():
        x1, y1, x2, y2, _, class_idx = map(int, obj)
        c = class_list[class_idx]

        for area_num, area_coords in areas.items():
            results = cv2.pointPolygonTest(np.array(area_coords, np.int32), ((x1 + x2) // 2, (y1 + y2) // 2), False)
            if results >= 0:
                area_objects[area_num].append(c)
                object_counts[area_num] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 0, 255), -1)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    total_objects = list(area_objects.items())[-1][1]
    total_cars = total_objects.count('car') + total_objects.count('motorcycle') + total_objects.count('cellphone')
    print(total_cars)
    space = (len(areas) - 1) - total_cars
    print(space)

    for area_num, count in object_counts.items():
        if count == 1:
            cv2.polylines(frame, [np.array(areas[area_num], np.int32)], True, (0, 0, 255), 2)
            cv2.putText(frame, str(area_num), (areas[area_num][0][0], areas[area_num][0][1] - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.polylines(frame, [np.array(areas[area_num], np.int32)], True, (0, 255, 0), 2)
            cv2.putText(frame, str(area_num), (areas[area_num][0][0], areas[area_num][0][1] - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Total Cars: {total_cars}", (23, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.putText(frame, f"Empty Slots: {str(space)}", (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("Parking Camera", frame)

    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
