from ultralytics import YOLO
from enum import Enum
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np


class ShootingHand(Enum):
    RIGHT = 0
    LEFT = 1

class MakeOrMissShot(Enum):
    MAKE = 0
    MISS = 1

hoop_model = YOLO("detect/train-6/weights/best.pt")
ball_model = YOLO("yolov8n.pt")
model = YOLO("yolo26n-pose.pt")

if __name__ == "__main__":
    pose_results = model("test.mp4", stream=True)
    hoop_results = hoop_model("test.mp4", stream=True)
    ball_results = ball_model("test.mp4", stream=True)


    object_dict = {}
    frame_dict = {}
    ball_pos_dict = {}
    wrist_pos_dict = {}
    arm_pos_dict = {}
    shoulder_pos_dict = {}

    for frame_num, result in enumerate(pose_results):
        frame_dict[frame_num] = result

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue

        k = result.keypoints.xy[0]

        left_wrist = k[9].tolist()
        right_wrist = k[10].tolist()

        wrist_pos_dict[frame_num] = {
            "left": left_wrist,
            "right": right_wrist
        }

        arm_pos_dict[frame_num] = {
            "left_elbow": k[7].tolist(),
            "left_wrist": left_wrist,
            "right_elbow": k[8].tolist(),
            "right_wrist": right_wrist
        }

        shoulder_pos_dict[frame_num] = {
            "left": k[5].tolist(),
            "right": k[6].tolist()
        }

    for frame_num, result in enumerate(hoop_results):
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            confidence = float(box.conf[0])

            object_dict[name] = (class_id, confidence)

    for frame_num, result in enumerate(ball_results):
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]

            if name == "sports ball":
                x, y, w, h = box.xywh[0].tolist()
                ball_pos_dict[frame_num] = [x, y]
                break


def elbow_degree(wrist: tuple, shoulder: tuple):
    delta_x = wrist[0] - shoulder[0]
    delta_y = wrist[1] - shoulder[1]

    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def distance(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    return np.linalg.norm(p1 - p2)


def find_release_frame(ball_pos_dict, wrist_pos_dict, shoulder_pos_dict):
    previous_distance = None

    for frame in sorted(ball_pos_dict.keys()):
        if frame not in wrist_pos_dict or ball_pos_dict[frame][1] > shoulder_pos_dict[frame]["left"][1]:
            continue

        ball = ball_pos_dict[frame]

        left_wrist = wrist_pos_dict[frame]["left"]
        right_wrist = wrist_pos_dict[frame]["right"]

        left_distance = distance(ball, left_wrist)
        right_distance = distance(ball, right_wrist)

        current_distance = min(left_distance, right_distance)

        if previous_distance is not None:
            if current_distance - previous_distance > 10:
                return frame
        previous_distance = current_distance

    return None

# shooting hand needs to be higher than guide hand when releasing (standard technique) or else this won't work as intended
def shooting_hand(release_frame, wrist_pos_dict):
    return ShootingHand.LEFT if wrist_pos_dict[release_frame]["left"][1] < wrist_pos_dict[release_frame]["right"][1] else ShootingHand.RIGHT


def ball_trajectory_arc(release_frame, ball_pos_dict):
    positions = [
        ball_pos_dict[frame]
        for frame in sorted(ball_pos_dict.keys())
        if frame >= release_frame
    ]

    if len(positions) < 3:
        raise ValueError("Need at least 3 ball positions after release")

    X = np.array(
        [p[0] for p in positions]
    ).reshape(-1, 1)

    y = np.array(
        [p[1] for p in positions]
    )

    model = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    )

    model.fit(X, y)

    regression = model.named_steps["linearregression"]

    b = regression.coef_[1]
    a = regression.coef_[2]

    release_pos_x = ball_pos_dict[release_frame][0]

    slope = 2 * a * release_pos_x + b

    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def make_or_miss_shot():
    for frames in ball_pos_dict:
        if ball_pos_dict[frames][1] != hoop_results[0][1]:
            continue
        
def hoop_center(box):
    x1, y1, x2, y2 = box.xyxy[0]

    hoop_x = ((x1 + x2) / 2).item()
    hoop_y = ((y1 + y2) / 2).item()

    return hoop_x, hoop_y

def dist_bw_ball_and_hoop(person, hoop_x):
    left_ankle = person[15]
    right_ankle = person[16]

    player_x = (left_ankle[0] + right_ankle[0]) / 2

    distance = hoop_x - player_x

    return distance

def optimal_shooting_arc(dist, release_height_m):
    hoop_height_m = 3.048

    height_diff = hoop_height_m - release_height_m

    optimal_angle = math.degrees(
        math.pi / 4 + 0.5 * math.atan2(height_diff, dist)
    )

    return optimal_angle

def shot_made(release_frame, ball_pos_dict, hoop_pos_dict):
    frames = sorted(f for f in ball_pos_dict if f > release_frame)

    for previous_frame, current_frame in zip(frames, frames[1:]):
        if current_frame - previous_frame > 2:
            continue

        hoop_frame = min(
            hoop_pos_dict,
            key=lambda f: abs(f - current_frame)
        )

        hx, hy, hw, hh = hoop_pos_dict[hoop_frame]

        x1, y1 = ball_pos_dict[previous_frame]
        x2, y2 = ball_pos_dict[current_frame]

        rim_y = hy

        if y1 < rim_y <= y2:
            t = (rim_y - y1) / (y2 - y1)
            crossing_x = x1 + t * (x2 - x1)

            if hx - hw / 2 < crossing_x < hx + hw / 2:
                return True

    return False

def pixels_to_meters(distance_pixels, hoop_y, ground_y):
    hoop_height_pixels = ground_y - hoop_y

    if hoop_height_pixels <= 0:
        raise ValueError("Invalid hoop-to-ground pixel height")

    meters_per_pixel = 3.048 / hoop_height_pixels

    return distance_pixels * meters_per_pixel

def find_shooting_hand(release_frame, ball_pos_dict, wrist_pos_dict):
    return ShootingHand.LEFT if distance(wrist_pos_dict[release_frame]["left"], ball_pos_dict[release_frame]) < distance(wrist_pos_dict[release_frame]["right"], ball_pos_dict[release_frame]) else ShootingHand.RIGHT

def elbow_angle(release_keypt, shooting_hand):

    if release_keypt is None or len(release_keypt) == 0:
        return None
    for partA, partB in ((6, 8), (8, 10), (5, 7), (7, 9)):
        match shooting_hand:
            case ShootingHand.RIGHT:
                if (partA, partB) == (6, 8):  # Right shoulder -> elbow
                    pointA = tuple(map(int, release_keypt[partA]))
                    pointB = tuple(map(int, release_keypt[partB]))

                elif (partA, partB) == (8, 10):  # Right elbow -> wrist
                    pointAs = tuple(map(int, release_keypt[partA]))
                    pointBs = tuple(map(int, release_keypt[partB]))
            case ShootingHand.LEFT:
                if (partA, partB) == (5, 7):  # Left shoulder -> elbow
                    pointA = tuple(map(int, release_keypt[partA]))
                    pointB = tuple(map(int, release_keypt[partB]))

                elif (partA, partB) == (7, 9):  # Left elbow -> wrist
                    pointAs = tuple(map(int, release_keypt[partA]))
                    pointBs = tuple(map(int, release_keypt[partB]))

    # Vector from elbow to shoulder
    vectorA = (pointA[0] - pointB[0],
            pointA[1] - pointB[1])

    # Vector from elbow to wrist
    vectorB = (pointBs[0] - pointAs[0],
            pointBs[1] - pointAs[1])

    dot = vectorA[0] * vectorB[0] + vectorA[1] * vectorB[1]

    magA = math.hypot(*vectorA)
    magB = math.hypot(*vectorB)

    if magA == 0 or magB == 0:
        return None

    cos_theta = dot / (magA * magB)

    angle = math.degrees(
        math.acos(max(-1, min(1, cos_theta)))
    )

    return angle
    


 

