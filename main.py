import cv2
import face_recognition
import numpy as np
import os
import time
import threading
import paho.mqtt.client as mqtt
import collections
import json
import pickle

# Đường dẫn tệp tin và thư mục
ENCODINGS_PATH = '/home/pi/FaceRecognitionWeb/encodings1.pkl'
UNKNOWN_FACES_PATH = '/home/pi/FaceRecognitionWeb/unknown_faces'

# Tạo thư mục lưu ảnh người lạ nếu chưa tồn tại
os.makedirs(UNKNOWN_FACES_PATH, exist_ok=True)

# Tải dữ liệu mã hóa khuôn mặt
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        encode_list_known = data['encodings']
        class_names = data['classNames']
    print('Dữ liệu mã hóa khuôn mặt đã được tải thành công.')
else:
    print(f"Lỗi: Không tìm thấy file {ENCODINGS_PATH}. Vui lòng kiểm tra lại.")
    exit()

# Biến trạng thái
door_status = "Không mở cửa"
running = True
recognition_enabled = True
saved_unknown_face = False
waiting_for_command = False
locked = False

# Cấu hình MQTT
MQTT_BROKER = "897e4e4bd28b411ba2464a4019281121.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC_STATUS = "homePod/6332f122-db2a-4ad2-9792-7998524fdaf5/faceRecognize"
MQTT_TOPIC_CONTROL = "homePod/6332f122-db2a-4ad2-9792-7998524fdaf5/doorStatus"

def on_connect(client, userdata, flags, rc):
    print(f'Kết nối MQTT với mã code: {rc}')

def on_message(client, userdata, msg):
    global recognition_enabled, waiting_for_command
    try:
        if msg.topic == MQTT_TOPIC_CONTROL:
            payload = msg.payload.decode('utf-8').strip()
        
            # Sửa lỗi dấu ngoặc kép không hợp lệ
            corrected_payload = payload.replace('“', '"').replace('”', '"')
            
            # Giải mã JSON
            data = json.loads(corrected_payload)

            print(data)
            
            # Lấy giá trị của "door"
            door_value = data.get("door", None)
            face_signal = door_value
            if face_signal == 0:
                print("Nhận lệnh bắt đầu nhận diện khuôn mặt.")
                recognition_enabled = True
                waiting_for_command = False
            elif face_signal == 1:
                print("Nhận lệnh dừng nhận diện khuôn mặt.")
                recognition_enabled = False
            else:
                print(f"Lệnh không hợp lệ: {face_signal}")
    except Exception as e:
        print(f"Lỗi xử lý tin nhắn MQTT: {e}")

# Cấu hình MQTT client
client = mqtt.Client()
client.tls_set()
client.username_pw_set("my_mqtt", "hellomqtt")
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)

def send_signal_to_shell(name):
    global door_status, waiting_for_command, locked
    if not locked:
        door_status = "Mở cửa" if name != "Người lạ" else "Không mở cửa"
        payload = json.dumps({"face":1 if name != "Người lạ" else 0})
        client.publish(MQTT_TOPIC_STATUS, payload)
        print(door_status)
        if name != "Người lạ":
            waiting_for_command = True

def recognize_faces():
    global running, recognition_enabled, waiting_for_command, saved_unknown_face, locked
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.")
        return

    frame_interval = 5
    frame_count = 0
    recognition_results = []
    start_time = time.time()
    recognition_interval = 5

    while running:
        if waiting_for_command or locked:
            time.sleep(0.5)
            saved_unknown_face = False
            recognition_results.clear()
            continue

        success, img = cap.read()
        if not success:
            print("Lỗi: Không thể đọc hình ảnh từ camera.")
            break

        if frame_count % frame_interval == 0:
            imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faces_cur_frame = face_recognition.face_locations(imgS, model='hog')
            encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)

            for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                matches = face_recognition.compare_faces(encode_list_known, encode_face, tolerance=0.5)
                face_dis = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_dis) if face_dis.size > 0 else -1

                if match_index != -1 and matches[match_index]:
                    name = class_names[match_index].upper()
                else:
                    name = "Người lạ"
                    if not saved_unknown_face:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        file_path = os.path.join(UNKNOWN_FACES_PATH, f"{timestamp}.jpg")
                        cv2.imwrite(file_path, img)
                        print(f"Lưu ảnh người lạ: {file_path}")
                        saved_unknown_face = True

                recognition_results.append(name)

        if time.time() - start_time >= recognition_interval:
            if recognition_results:
                most_common_name = collections.Counter(recognition_results).most_common(1)[0][0]
                print(f"Kết quả nhận diện: {most_common_name}")

                if most_common_name == "Người lạ":
                    locked = True
                    print("Khóa nhận diện trong 30 giây...")
                    time.sleep(10)
                    locked = False
                else:
                    send_signal_to_shell(most_common_name)

            recognition_results.clear()
            start_time = time.time()
            saved_unknown_face = False

        frame_count += 1

    cap.release()
    print("Dừng nhận diện khuôn mặt.")

def start_face_recognition():
    global running
    running = True
    threading.Thread(target=recognize_faces, daemon=True).start()

def stop_face_recognition():
    global running
    running = False

try:
    client.subscribe(MQTT_TOPIC_CONTROL)
    client.loop_start()
    start_face_recognition()

    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Dừng chương trình...")
    stop_face_recognition()
    client.loop_stop()
