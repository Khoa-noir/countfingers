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
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = (v1[0]**2 + v1[1]**2)**0.5
    mag2 = (v2[0]**2 + v2[1]**2)**0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
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
st.title("Nhận diện cử chỉ bàn tay")

# ----------------- Sidebar -----------------
camera_on = st.sidebar.checkbox("Bật Camera")
set_new_password = st.sidebar.checkbox("Đặt mật khẩu cử chỉ")
unlock_by_gesture = st.sidebar.checkbox("Mở khóa bằng cử chỉ")

counting_number = st.sidebar.checkbox("Hiển thị số ngón")
display_landmarks = st.sidebar.checkbox("Hiển thị landmark")
show_features = st.sidebar.checkbox("Hiển thị giá trị đặc trưng")

# ----------------- Session state -----------------
if 'gesture_password' not in st.session_state:
    st.session_state.gesture_password = None
if 'record' not in st.session_state:
    st.session_state.record = False
if 'last_unlock_time' not in st.session_state:
    st.session_state.last_unlock_time = 0
if 'last_wrong_time' not in st.session_state:
    st.session_state.last_wrong_time = 0

# ----------------- Nút ghi mật khẩu -----------------
if set_new_password:
    if st.sidebar.button("Ghi lại mật khẩu"):
        st.session_state.record = True
        st.success("Hãy đưa tay vào khung camera để ghi mật khẩu!")

# ----------------- Camera input -----------------
if camera_on:
    img_file = st.camera_input("Camera", key="cam")

    if img_file:
        img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        gesture_info = None
        total_fingers = 0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                if display_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture_info = extract_gesture_features(hand_landmarks, img.shape)

                if counting_number:
                    total_fingers = gesture_info['finger_count']
                    cv2.putText(img, f"{total_fingers} fingers", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Record password
                if st.session_state.record:
                    st.session_state.gesture_password = gesture_info
                    st.session_state.record = False
                    st.success("Đã lưu mật khẩu cử chỉ!")

                # Unlock
                if unlock_by_gesture and st.session_state.gesture_password:
                    now = time.time()
                    if compare_gestures(gesture_info, st.session_state.gesture_password):
                        st.session_state.last_unlock_time = now
                    else:
                        st.session_state.last_wrong_time = now

                # Show feature values
                if show_features and gesture_info:
                    y = 50
                    for key, v in gesture_info.items():
                        txt = f"{key}: {v:.2f}" if isinstance(v, float) else f"{key}: {v}"
                        cv2.putText(img, txt, (400, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        y += 20

        if time.time() - st.session_state.last_unlock_time < 3:
            cv2.putText(img, "ĐÃ MỞ KHÓA", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        if time.time() - st.session_state.last_wrong_time < 3:
            cv2.putText(img, "CỬ CHỈ SAI", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

else:
    st.info("Bật camera để bắt đầu.")

# ".conda/python.exe" -m streamlit run source.py


