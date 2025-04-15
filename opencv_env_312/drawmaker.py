import cv2

# 使用するマーカーディクショナリ
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# ArUcoマーカー4枚生成（サイズ: 200×200）
for i in range(4):
    img = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
    cv2.imwrite(f"marker_{i}.png", img)

print("✅ marker_0.png ～ marker_3.png を出力しました")
