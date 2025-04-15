import cv2
import numpy as np
import csv
import tkinter as tk
import os

def btn_click():
    try:
        x_dis = int(txt_1.get())
        y_dis = int(txt_2.get())
        cap_num = int(txt_3.get())
    except ValueError:
        print("数字を入力してください")
        return

    root.destroy()

    size = 3
    cap = cv2.VideoCapture(cap_num)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    cv2.namedWindow('binary')
    cv2.createTrackbar('threshold', 'binary', 90, 256, lambda x: None)

    log = []
    frame_count = 0

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("カメラエラー")
                break

            frame_count += 1

            corners, ids, _ = detector.detectMarkers(img)

            if ids is None or len(ids) < 4:
                cv2.imshow('raw', img)
                cv2.waitKey(1)
                continue

            corners2 = [None] * 4  # マーカーID 0～3 に対応
            for i, c in zip(ids.ravel(), corners):
                if 0 <= i <= 3:
                    corners2[i] = c.copy()

            if any(c is None for c in corners2):
                print("ID 0〜3 のマーカーが揃っていません")
                cv2.imshow('raw', img)
                cv2.waitKey(1)
                continue

            m = np.empty((4, 2))
            m[0] = corners2[0][0][2]
            m[1] = corners2[1][0][3]
            m[2] = corners2[2][0][0]
            m[3] = corners2[3][0][1]

            width, height = (x_dis * size, y_dis * size)
            marker_coordinates = np.float32(m)
            true_coordinates = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            trans_mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
            img_trans = cv2.warpPerspective(img, trans_mat, (width, height))
            tmp = img_trans.copy()
            img_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)

            th_val = cv2.getTrackbarPos('threshold', 'binary')
            _, img_trans = cv2.threshold(img_trans, th_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(img_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                cv2.imshow('raw', img)
                cv2.waitKey(1)
                continue

            rect = cv2.minAreaRect(contours[0])
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle += 90
            angle = angle % 180  # 0～180度に制限

            box = cv2.boxPoints(rect)
            box = np.intp(box)
            tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 2)

            mu = cv2.moments(img_trans, False)
            if mu["m00"] != 0:
                x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])
            else:
                cv2.imshow('raw', img)
                cv2.waitKey(1)
                continue

            cv2.putText(tmp, f"Angle: {angle:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('image', tmp)

            log.append([frame_count, x, y, angle])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        csv_filename = "sashigane_angle_log.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "center_x", "center_y", "angle_deg"])
            writer.writerows(log)
        print(f"CSV保存完了: {os.path.abspath(csv_filename)}")

# === GUI ===
root = tk.Tk()
root.title('さしがね角度計測')
root.geometry('300x200')

lbl_width = tk.Label(root, text='幅')
lbl_width.place(x=30, y=70)
txt_1 = tk.Entry(root, width=20)
txt_1.insert(0, "145")
txt_1.place(x=90, y=70)

lbl_height = tk.Label(root, text='高さ')
lbl_height.place(x=30, y=100)
txt_2 = tk.Entry(root, width=20)
txt_2.insert(0, "145")
txt_2.place(x=90, y=100)

lbl_camera = tk.Label(root, text='カメラ番号')
lbl_camera.place(x=30, y=150)
txt_3 = tk.Entry(root, width=20)
txt_3.insert(0, "0")
txt_3.place(x=90, y=150)

btn = tk.Button(root, text='OK', command=btn_click)
btn.place(x=140, y=170)

root.mainloop()
