import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import math

# ----------------- Khởi tạo MediaPipe Hands -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------- Hàm xử lý -----------------
def landmarks_distance(a, b, img_shape):
    h, w, _ = img_shape
    xa, ya = int(a.x * w), int(a.y * h)
    xb, yb = int(b.x * w), int(b.y * h)
    return np.sqrt((xa - xb)**2 + (ya - yb)**2)

def number_identify(hand_landmarks, img_shape):
    lm = hand_landmarks.landmark
    count = 0
    if landmarks_distance(lm[4], lm[17], img_shape) > landmarks_distance(lm[2], lm[17], img_shape):
        count += 1
    if landmarks_distance(lm[8], lm[0], img_shape) > landmarks_distance(lm[5], lm[0], img_shape):
        count += 1
    if landmarks_distance(lm[12], lm[0], img_shape) > landmarks_distance(lm[9], lm[0], img_shape):
        count += 1
    if landmarks_distance(lm[16], lm[0], img_shape) > landmarks_distance(lm[13], lm[0], img_shape):
        count += 1
    if landmarks_distance(lm[20], lm[0], img_shape) > landmarks_distance(lm[17], lm[0], img_shape):
        count += 1
    return count

def landmarks_angular(lm1, lm2, lm3):
    v1 = (lm1.x - lm2.x, lm1.y - lm2.y)
    v2 = (lm3.x - lm2.x, lm3.y - lm2.y)
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    mag_v1 = (v1[0]**2 + v1[1]**2) ** 0.5
    mag_v2 = (v2[0]**2 + v2[1]**2) ** 0.5
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0
    cos_angle = max(min(dot_product / (mag_v1 * mag_v2), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def extract_gesture_features(hand_landmarks, img_shape):
    lm = hand_landmarks.landmark
    return {
        'finger_count': number_identify(hand_landmarks, img_shape),
        'thumb_angle': landmarks_angular(lm[0], lm[2], lm[4]),
        'index_angle': landmarks_angular(lm[5], lm[6], lm[8]),
        'middle_angle': landmarks_angular(lm[9], lm[10], lm[12]),
        'ring_angle': landmarks_angular(lm[13], lm[14], lm[16]),
        'pinky_angle': landmarks_angular(lm[17], lm[18], lm[20]),
        'thumb_index_angle': landmarks_angular(lm[4], lm[0], lm[8]),
        'index_middle_angle': landmarks_angular(lm[8], lm[0], lm[12]),
        'middle_ring_angle': landmarks_angular(lm[12], lm[0], lm[16]),
        'ring_pinky_angle': landmarks_angular(lm[16], lm[0], lm[20])
    }

def compare_gestures(current, saved, tolerance=0.5):
    for key in saved:
        if key == 'finger_count':
            if current[key] != saved[key]:
                return False
        else:
            low = saved[key] * (1 - tolerance)
            high = saved[key] * (1 + tolerance)
            if not (low <= current[key] <= high):
                return False
    return True

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Hand Gesture Unlock", layout="wide")

st.markdown("<h1 style='text-align:center'>Nhận diện cử chỉ bàn tay</h1>", unsafe_allow_html=True)

# Hướng dẫn sử dụng
st.markdown("""
### Hướng dẫn sử dụng:
1. Bật camera.
2. (Tùy chọn) Đặt mật khẩu cử chỉ: Tick và nhấn 'Record Gesture Password'.
3. (Tùy chọn) Bật mở khóa bằng cử chỉ: Tick 'Unlock by Gesture'.
4. Xem các tính năng hiển thị: số ngón, landmark, giá trị đặc trưng.
""")

# --------- Nhóm điều khiển ----------
st.sidebar.header("Bước 1: Điều khiển chính")
camera_on = st.sidebar.checkbox("Bật Camera")
set_new_password = st.sidebar.checkbox("Đặt mật khẩu cử chỉ")
unlock_by_gesture = st.sidebar.checkbox("Mở khóa bằng cử chỉ")

if set_new_password:
    if st.sidebar.button("Ghi lại mật khẩu"):
        st.session_state.record_gesture = True
        st.success("Sẵn sàng ghi lại cử chỉ! Thực hiện cử chỉ của bạn.")

# --------- Nhóm hiển thị video & tính năng ----------
st.sidebar.header("Bước 2: Hiển thị")
counting_number = st.sidebar.checkbox("Hiển thị số ngón")
display_landmarks = st.sidebar.checkbox("Hiển thị landmark")
show_features = st.sidebar.checkbox("Hiển thị giá trị đặc trưng")

# Video frame
FRAME_WINDOW = st.image([])

# Khởi tạo session state
if 'gesture_password' not in st.session_state:
    st.session_state.gesture_password = None
if 'record_gesture' not in st.session_state:
    st.session_state.record_gesture = False
if 'last_unlock_time' not in st.session_state:
    st.session_state.last_unlock_time = 0
if 'last_wrong_time' not in st.session_state:
    st.session_state.last_wrong_time = 0

# ----------------- Xử lý video -----------------
if camera_on:
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập camera")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        total_fingers = 0
        gesture_info = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Hiển thị số ngón
                if counting_number:
                    fingers = number_identify(hand_landmarks, frame.shape)
                    total_fingers += fingers
                    cv2.putText(frame, f"Số ngón: {total_fingers}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Vẽ landmark
                if display_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Trích xuất đặc trưng
                gesture_info = extract_gesture_features(hand_landmarks, frame.shape)
                # Ghi lại mật khẩu
                if st.session_state.record_gesture:
                    st.session_state.gesture_password = gesture_info
                    st.session_state.record_gesture = False
                    st.success("Đã lưu mật khẩu cử chỉ!")
                    st.write("Đặc trưng đã lưu:", st.session_state.gesture_password)
                # Mở khóa
                current_time = time.time()
                if unlock_by_gesture and st.session_state.gesture_password and (current_time - st.session_state.last_unlock_time) > 3:
                    if compare_gestures(gesture_info, st.session_state.gesture_password):
                        st.session_state.last_unlock_time = current_time
                    else:
                        st.session_state.last_wrong_time = current_time
                # Hiển thị giá trị đặc trưng
                if show_features and gesture_info:
                    y_pos = 10
                    for key, val in gesture_info.items():
                        text = f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}"
                        cv2.putText(frame, text, (450, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        y_pos += 15

        # Thông báo mở khóa
        if (time.time() - st.session_state.last_unlock_time) <= 3:
            cv2.putText(frame, "ĐÃ MỞ KHÓA", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # Thông báo cử chỉ sai
        if (time.time() - st.session_state.last_wrong_time) <= 3:
            cv2.putText(frame, "CỬ CHỈ KHÔNG ĐÚNG", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.info("Camera đang tắt. Bật 'Bật Camera' để bắt đầu.")



# ".conda/python.exe" -m streamlit run source.py
