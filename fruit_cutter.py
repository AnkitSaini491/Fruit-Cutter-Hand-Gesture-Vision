
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math


# ==========================================
# SETTINGS
# ==========================================

WIDTH = 1000
HEIGHT = 700

GAME_TIME = 60

FRUIT_TYPES = [
    ("🍎", (0, 0, 255)),
    ("🍊", (0, 140, 255)),
    ("🍉", (0, 200, 0)),
    ("🍋", (0, 255, 255)),
    ("🍇", (150, 0, 150))
]


# ==========================================
# MEDIAPIPE HAND
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# ==========================================
# GAME VARIABLES
# ==========================================

score = 0
lives = 3

start_time = time.time()

fruits = []

last_spawn = 0

trail = []


# ==========================================
# CREATE FRUIT
# ==========================================

def create_fruit():

    fruit = random.choice(FRUIT_TYPES)

    return {
        "x": random.randint(80, WIDTH - 80),
        "y": HEIGHT + 50,
        "speed": random.randint(7, 12),
        "radius": random.randint(30, 45),
        "symbol": fruit[0],
        "color": fruit[1]
    }


# ==========================================
# DISTANCE FUNCTION
# ==========================================

def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    # ======================================
    # TIMER
    # ======================================

    elapsed = int(time.time() - start_time)

    remaining = max(
        0,
        GAME_TIME - elapsed
    )


    # ======================================
    # SPAWN FRUITS
    # ======================================

    current_time = time.time()

    if current_time - last_spawn > 0.8:

        fruits.append(
            create_fruit()
        )

        last_spawn = current_time


    # ======================================
    # HAND POSITION
    # ======================================

    hand_x = None
    hand_y = None

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # INDEX FINGER TIP
        landmark = hand.landmark[8]

        hand_x = int(
            landmark.x * WIDTH
        )

        hand_y = int(
            landmark.y * HEIGHT
        )


        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # Draw fingertip
        cv2.circle(
            frame,
            (hand_x, hand_y),
            12,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (hand_x, hand_y),
            18,
            (0, 255, 255),
            2
        )


        # ==================================
        # CUTTING TRAIL
        # ==================================

        trail.append(
            (hand_x, hand_y)
        )

        if len(trail) > 15:
            trail.pop(0)


    # ======================================
    # DRAW CUTTING TRAIL
    # ======================================

    for i in range(1, len(trail)):

        cv2.line(
            frame,
            trail[i - 1],
            trail[i],
            (255, 255, 0),
            3
        )


    # ======================================
    # UPDATE FRUITS
    # ======================================

    new_fruits = []

    for fruit in fruits:

        fruit["y"] -= fruit["speed"]


        # Draw fruit
        cv2.circle(
            frame,
            (
                fruit["x"],
                fruit["y"]
            ),
            fruit["radius"],
            fruit["color"],
            -1
        )


        # Fruit highlight
        cv2.circle(
            frame,
            (
                fruit["x"] - 10,
                fruit["y"] - 10
            ),
            6,
            (255, 255, 255),
            -1
        )


        # ==================================
        # FRUIT CUT DETECTION
        # ==================================

        if hand_x is not None:

            d = distance(
                hand_x,
                hand_y,
                fruit["x"],
                fruit["y"]
            )

            if d < fruit["radius"] + 25:

                score += 10

                # Cut effect
                cv2.line(
                    frame,
                    (
                        fruit["x"] - 35,
                        fruit["y"] - 35
                    ),
                    (
                        fruit["x"] + 35,
                        fruit["y"] + 35
                    ),
                    (255, 255, 255),
                    5
                )

                continue


        # ==================================
        # MISSED FRUIT
        # ==================================

        if fruit["y"] < -50:

            lives -= 1

        else:

            new_fruits.append(
                fruit
            )


    fruits = new_fruits


    # ======================================
    # HEADER
    # ======================================

    cv2.rectangle(
        frame,
        (0, 0),
        (WIDTH, 80),
        (20, 20, 20),
        -1
    )


    cv2.putText(
        frame,
        f"SCORE: {score}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )


    cv2.putText(
        frame,
        f"LIVES: {lives}",
        (350, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 100, 255),
        3
    )


    cv2.putText(
        frame,
        f"TIME: {remaining}",
        (700, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 0),
        3
    )


    # ======================================
    # GAME OVER
    # ======================================

    if remaining <= 0 or lives <= 0:

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (200, 200),
            (800, 500),
            (20, 20, 20),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.85,
            frame,
            0.15,
            0
        )

        cv2.putText(
            frame,
            "GAME OVER",
            (350, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

        cv2.putText(
            frame,
            f"FINAL SCORE: {score}",
            (325, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "Press R to Restart",
            (335, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (355, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Fruit Cutter - Hand Vision",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # QUIT
    if key == ord("q"):

        break


    # RESTART
    if key == ord("r"):

        score = 0
        lives = 3
        fruits = []
        trail = []

        start_time = time.time()


# ==========================================
# RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()
