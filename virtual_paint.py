import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# Capture Video
cap = cv2.VideoCapture(1)

# Set FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Create window and set to fullscreen
cv2.namedWindow("Virtual Paint", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Paint", 1000, 800)  # Example size

# Create initial canvas
canvas = None

# Define Colors and Brush Settings
colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]
color_index = 0
prev_x, prev_y = None, None

# Cooldown timers for actions
last_color_change = 0
last_clear = 0
cooldown = 0.5  # seconds

# Performance monitoring
prev_time = 0
fps_list = []


def calculate_fps():
    global prev_time, fps_list
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    fps_list.append(fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    return sum(fps_list) / len(fps_list)


def calculate_ui_parameters(frame):
    """Calculate UI parameters based on frame dimensions"""
    height, width = frame.shape[:2]
    min_dim = min(width, height)

    # Calculate brush and eraser sizes relative to frame
    brush_size = max(1, int(min_dim * 0.005))  # 0.5% of min dimension
    eraser_size = max(5, int(min_dim * 0.04))  # 4% of min dimension

    # Calculate palette dimensions
    palette_size = int(min_dim * 0.08)  # 8% of min dimension
    palette_gap = int(palette_size * 0.25)  # 25% of palette size
    palette_x = int(width * 0.02)  # 2% from left
    palette_y = int(height * 0.02)  # 2% from top

    # Calculate font parameters
    font_scale = min_dim / 1000  # Scale font based on frame size
    text_start_y = int(height * 0.80)  # 80% down
    text_line_height = int(height * 0.04)  # 4% of height

    return {
        "brush_size": brush_size,
        "eraser_size": eraser_size,
        "palette_size": palette_size,
        "palette_gap": palette_gap,
        "palette_x": palette_x,
        "palette_y": palette_y,
        "font_scale": font_scale,
        "text_start_y": text_start_y,
        "text_line_height": text_line_height,
    }


def draw_eraser_cursor(frame, x, y, size):
    """Draw eraser cursor with crosshair"""
    # Draw outer circle
    cv2.circle(frame, (x, y), size, (255, 255, 255), 2)
    # Draw crosshair
    line_length = size // 2
    cv2.line(frame, (x - line_length, y), (x + line_length, y), (255, 255, 255), 1)
    cv2.line(frame, (x, y - line_length), (x, y + line_length), (255, 255, 255), 1)
    # Draw small center circle
    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


def is_finger_near_palette(x, y, ui_params):
    """Check if finger is in palette area with dynamic positioning"""
    for i in range(len(colors)):
        palette_rect_x = ui_params["palette_x"] + i * (
            ui_params["palette_size"] + ui_params["palette_gap"]
        )
        if (
            palette_rect_x <= x <= palette_rect_x + ui_params["palette_size"]
            and ui_params["palette_y"]
            <= y
            <= ui_params["palette_y"] + ui_params["palette_size"]
        ):
            return i
    return -1


quit_requested = False

# Get initial frame to setup canvas
ret, frame = cap.read()
if ret:
    frame = cv2.flip(frame, 1)
    canvas = np.zeros(frame.shape, dtype=np.uint8)

while cap.isOpened() and not quit_requested:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Calculate UI parameters for current frame
    ui_params = calculate_ui_parameters(frame)

    # Convert to RGB and process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    result = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    current_time = time.time()
    drawing_mode = False
    eraser_mode = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]
            ring_finger_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Convert coordinates
            x = int(index_finger_tip.x * frame.shape[1])
            y = int(index_finger_tip.y * frame.shape[0])

            # Check finger positions
            index_up = index_finger_tip.y < hand_landmarks.landmark[6].y
            middle_up = middle_finger_tip.y < hand_landmarks.landmark[10].y
            ring_up = ring_finger_tip.y < hand_landmarks.landmark[14].y
            pinky_up = pinky_tip.y < hand_landmarks.landmark[18].y

            # Quit gesture: index and pinky up, others down
            if index_up and not middle_up and not ring_up and pinky_up:
                quit_requested = True
                break

            # Drawing mode: index finger up, others down
            elif index_up and not middle_up and not ring_up and not pinky_up:
                drawing_mode = True
                if prev_x is not None and prev_y is not None:
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        colors[color_index],
                        ui_params["brush_size"],
                        cv2.LINE_AA,
                    )
                prev_x, prev_y = x, y

            # Color selection mode: index and middle fingers up
            elif index_up and middle_up and not ring_up and not pinky_up:
                prev_x, prev_y = None, None
                palette_selection = is_finger_near_palette(x, y, ui_params)
                if (
                    palette_selection >= 0
                    and current_time - last_color_change > cooldown
                ):
                    color_index = palette_selection
                    last_color_change = current_time
                cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)

            # Clear canvas: first three fingers up
            elif index_up and middle_up and ring_up and not pinky_up:
                if current_time - last_clear > cooldown:
                    canvas.fill(0)
                    last_clear = current_time
                prev_x, prev_y = None, None

            # Eraser mode: all fingers up
            elif index_up and middle_up and ring_up and pinky_up:
                eraser_mode = True
                if prev_x is not None and prev_y is not None:
                    cv2.circle(canvas, (x, y), ui_params["eraser_size"], (0, 0, 0), -1)
                prev_x, prev_y = x, y
                # Draw eraser cursor
                draw_eraser_cursor(frame, x, y, ui_params["eraser_size"])

            else:
                prev_x, prev_y = None, None

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=1, circle_radius=1),
            )
    else:
        prev_x, prev_y = None, None

    # Blend canvas with camera feed
    result_frame = frame.copy()
    mask = canvas > 0
    result_frame[mask] = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)[mask]

    # Draw color palette
    for i, color in enumerate(colors):
        palette_rect_x = ui_params["palette_x"] + i * (
            ui_params["palette_size"] + ui_params["palette_gap"]
        )
        cv2.rectangle(
            result_frame,
            (palette_rect_x, ui_params["palette_y"]),
            (
                palette_rect_x + ui_params["palette_size"],
                ui_params["palette_y"] + ui_params["palette_size"],
            ),
            color,
            -1,
        )
        if i == color_index:
            cv2.rectangle(
                result_frame,
                (palette_rect_x - 2, ui_params["palette_y"] - 2),
                (
                    palette_rect_x + ui_params["palette_size"] + 2,
                    ui_params["palette_y"] + ui_params["palette_size"] + 2,
                ),
                (255, 255, 255),
                2,
            )

    # Display instructions
    instructions = [
        "Index finger: Draw",
        "Index + Middle: Select color",
        "Three fingers: Clear",
        "Four/All Fingers: Erase",
        "Index + Pinky: Quit",
    ]

    for i, instruction in enumerate(instructions):
        cv2.putText(
            result_frame,
            instruction,
            (
                ui_params["palette_x"],
                ui_params["text_start_y"] + i * ui_params["text_line_height"],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            ui_params["font_scale"],
            (255, 255, 255),
            1,
        )

    # Show FPS
    fps = calculate_fps()
    fps_x = frame.shape[1] - int(frame.shape[1] * 0.2)
    fps_y = int(frame.shape[0] * 0.1)
    cv2.putText(
        result_frame,
        f"FPS: {fps:.1f}",
        (fps_x, fps_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        ui_params["font_scale"] * 1.4,
        (0, 255, 0),
        2,
    )

    # Display result
    cv2.imshow("Virtual Paint", result_frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
