from ultralytics import YOLO
import cv2
from datetime import datetime
import csv
import os
import time

model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counted_ids = set()
hourly_count = 0
daily_count = 0
previous_day_key = datetime.now().date()
previous_hour_key = datetime.now().strftime("%Y-%m-%d %H")

frame_count = 0
fps = 0
prev_time = time.time()

# CSVファイルがない場合は作成
if not os.path.exists("counts.csv"):  
    with open("counts.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Date Hour", "Visitor Count"])

if not cap.isOpened():
    print("Cannot open camera")
    exit()
# ---- 描画を間引く設定 ----
DRAW_EVERY_N = 3  # 3フレームに1回だけ plot() する
frame_idx = 0
last_drawn = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame_idx += 1
    # FPSを計算（1秒ごとに更新）
    frame_count += 1
    now_time = time.time()
    elapsed = now_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        prev_time = now_time
    # 現在の時刻取得
    now = datetime.now()
    day_key = now.strftime("%Y-%m-%d")
    hour_key = now.strftime("%Y-%m-%d %H")

    # 日付が変わったらカウントをリセット
    if day_key != previous_day_key:
        daily_count = 0
        counted_ids = set()
        previous_day_key = day_key

    # 時間が変わったらCSVに保存
    if hour_key != previous_hour_key:
        with open("counts.csv", mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([previous_hour_key, hourly_count])
        hourly_count = 0
        previous_hour_key = hour_key

    # トラッキング付き推論（追跡モード）
    results = model.track(frame, persist=True, classes=[0], imgsz=320, conf=0.45)

    # 描画フレーム生成
    annotated_frame = results[0].plot()
    
     # ---- plot()を間引き ----
    if frame_idx % DRAW_EVERY_N == 0:
        annotated_frame = results[0].plot()
        last_drawn = annotated_frame

    # ---- 最新の描画結果を表示 ----
    display_frame = last_drawn if last_drawn is not None else frame.copy()


    # カウント処理
    boxes = results[0].boxes
    if boxes.id is not None and boxes.cls is not None:
        track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        for track_id, cls_id in zip(track_ids, class_ids):
            if model.names[int(cls_id)] == "person":
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    hourly_count += 1
                    daily_count += 1


    # 累計表示
    cv2.putText(annotated_frame, f"{day_key} Total Visitors: {daily_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
    # FPS表示
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    


    cv2.imshow("Tracking Count", annotated_frame)


    if cv2.waitKey(1) & 0xFF in [27, ord('q')]: 
        break

cap.release()
cv2.destroyAllWindows()

