import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Получаем размеры экрана
screen_w, screen_h = pyautogui.size()

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

# Индексы ключевых точек для глаз
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Переменные для сглаживания
smooth_x, smooth_y = screen_w // 2, screen_h // 2
alpha = 0.3  # Коэффициент сглаживания

def get_eye_position(landmarks, eye_indices, iris_indices, frame_w, frame_h):
    """Вычисляет позицию взгляда для одного глаза"""
    # Получаем координаты радужки
    iris_center = np.mean([(landmarks[idx].x, landmarks[idx].y) 
                           for idx in iris_indices], axis=0)
    
    # Получаем границы глаза
    eye_points = [(landmarks[idx].x, landmarks[idx].y) 
                  for idx in eye_indices]
    
    # Находим центр глаза и его размеры
    eye_center = np.mean(eye_points, axis=0)
    eye_left = min([p[0] for p in eye_points])
    eye_right = max([p[0] for p in eye_points])
    eye_top = min([p[1] for p in eye_points])
    eye_bottom = max([p[1] for p in eye_points])
    
    # Нормализуем позицию радужки относительно глаза
    eye_width = eye_right - eye_left
    eye_height = eye_bottom - eye_top
    
    if eye_width > 0 and eye_height > 0:
        x_ratio = (iris_center[0] - eye_left) / eye_width
        y_ratio = (iris_center[1] - eye_top) / eye_height
        return x_ratio, y_ratio
    
    return 0.5, 0.5

print("=" * 60)
print("Eye Gaze Tracker - Отслеживание взгляда")
print("=" * 60)
print(f"Размер экрана: {screen_w}x{screen_h}")
print("\nИнструкции:")
print("- Смотрите на камеру")
print("- Двигайте глазами, чтобы увидеть координаты")
print("- Нажмите 'q' для выхода")
print("- Нажмите 'c' для калибровки (центр экрана)")
print("=" * 60)

# Параметры калибровки
offset_x, offset_y = 0, 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры")
        break
    
    # Отражаем кадр для естественного отображения
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    
    # Конвертируем в RGB для MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # Получаем позицию взгляда для обоих глаз
        left_x, left_y = get_eye_position(landmarks, LEFT_EYE, LEFT_IRIS, 
                                         frame_w, frame_h)
        right_x, right_y = get_eye_position(landmarks, RIGHT_EYE, RIGHT_IRIS, 
                                           frame_w, frame_h)
        
        # Усредняем результаты обоих глаз
        avg_x = (left_x + right_x) / 2
        avg_y = (left_y + right_y) / 2
        
        # Преобразуем в координаты экрана (с инверсией X для зеркального отображения)
        gaze_x = int((1 - avg_x) * screen_w) + offset_x
        gaze_y = int(avg_y * screen_h) + offset_y
        
        # Ограничиваем координаты размерами экрана
        gaze_x = max(0, min(screen_w, gaze_x))
        gaze_y = max(0, min(screen_h, gaze_y))
        
        # Сглаживание координат
        smooth_x = int(alpha * gaze_x + (1 - alpha) * smooth_x)
        smooth_y = int(alpha * gaze_y + (1 - alpha) * smooth_y)
        
        # Отрисовка глаз на видео
        for idx in LEFT_IRIS + RIGHT_IRIS:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Отображаем координаты на экране
        cv2.putText(frame, f"Gaze: ({smooth_x}, {smooth_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Screen: {screen_w}x{screen_h}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Вывод в консоль
        print(f"\rКоординаты взгляда: X={smooth_x:4d}, Y={smooth_y:4d}", 
              end="", flush=True)
    else:
        cv2.putText(frame, "Лицо не обнаружено", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Отображаем видео
    cv2.imshow('Eye Gaze Tracker (нажмите q для выхода)', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Калибровка - установить текущую позицию как центр экрана
        offset_x = (screen_w // 2) - smooth_x
        offset_y = (screen_h // 2) - smooth_y
        print("\n[Калибровка выполнена - центр установлен]")

cap.release()
cv2.destroyAllWindows()
print("\n\nПрограмма завершена.")