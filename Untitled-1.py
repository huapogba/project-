import cv2
import pygame
from ultralytics import YOLO

# Khởi tạo mô hình YOLOv8
model = YOLO("yolov8n.pt")

# Khởi tạo pygame để phát âm thanh
pygame.mixer.init()
sound = pygame.mixer.Sound(r"C:\Users\ADMIN\Desktop\1.wav")  # Thay bằng file âm thanh của bạn

# Mở webcam (hoặc thay bằng đường dẫn video)
cap = cv2.VideoCapture(0)
playing_sound = False  # Biến kiểm tra trạng thái âm thanh

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chạy nhận diện trên khung hình
    results = model(frame)
    person_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Nhãn lớp

            # Nếu là con người (class 0 trong COCO dataset)
            if cls == 0:
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Nếu phát hiện người và chưa phát âm thanh
    if person_detected and not playing_sound:
        sound.play()
        playing_sound = True  # Đánh dấu đang phát âm thanh

    # Nếu không phát hiện người, đặt lại biến
    if not person_detected and playing_sound:
        if sound.get_num_channels() == 0:  # Kiểm tra nếu âm thanh đã phát xong
            playing_sound = False

    # Hiển thị kết quả
    cv2.imshow("YOLOv8 - Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
