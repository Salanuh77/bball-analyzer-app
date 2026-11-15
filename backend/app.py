from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from typing import Literal
import shutil
from pydantic import BaseModel, Field
import analyze
import math
import asyncio
import cv2
import os
import requests

app = FastAPI()

release_frame = None

object_dict = {}
frame_dict = {}
ball_pos_dict = {}
wrist_pos_dict = {}
arm_pos_dict = {}
shoulder_pos_dict = {}
person_pos_dict = {}
hoop_pos_dict = {}

class User(BaseModel):
    id: int
    username: str
    curVid: str
    vids: list[str]

POSE_PAIRS = [
    (5, 6),    # Shoulders
    (5, 7),    # Left upper arm
    (7, 9),    # Left forearm
    (6, 8),    # Right upper arm
    (8, 10),   # Right forearm
    (5, 11),   # Left torso
    (6, 12),   # Right torso
    (11, 12),  # Hips
    (11, 13),  # Left thigh
    (13, 15),  # Left shin
    (12, 14),  # Right thigh
    (14, 16)   # Right shin
]

@app.post("/videos") 
async def analyze_videos(vids: UploadFile = File(...)):
    global release_frame
    if not vids.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File uploaded is not a video. Got: {vids.content_type}")
    
    with open("uploaded_video.mp4", "wb") as buffer:
        shutil.copyfileobj(vids.file, buffer)

    print("Saved size:", os.path.getsize("uploaded_video.mp4"))
    print("Uploaded filename:", vids.filename)

    cap = cv2.VideoCapture("uploaded_video.mp4")
    print("OpenCV opened:", cap.isOpened())
    cap.release()

    pose_results = analyze.model("uploaded_video.mp4", stream=True)
    hoop_results = analyze.hoop_model("uploaded_video.mp4",
        stream=True,
        imgsz=1280,
        conf=0.05
    )
    ball_results = analyze.ball_model("uploaded_video.mp4", stream=True)

    for frame_num, result in enumerate(pose_results):
        frame_dict[frame_num] = result

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue

        k = result.keypoints.xy[0]
        person_pos_dict[frame_num] = k

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

    print("MODEL CLASSES:", analyze.hoop_model.names)

    for frame_num, result in enumerate(hoop_results):
        print("Frame:", frame_num, "Detections:", len(result.boxes))

        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            confidence = float(box.conf[0])

            print("Class:", name, "Confidence:", confidence)

            if name == "hoops":
                x, y, w, h = box.xywh[0].tolist()
                hoop_pos_dict[frame_num] = (x, y, w, h)


    for frame_num, result in enumerate(ball_results):
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]

            if name == "sports ball":
                x, y, w, h = box.xywh[0].tolist()
                ball_pos_dict[frame_num] = [x, y]
                break

    release_frame = analyze.find_release_frame(ball_pos_dict, wrist_pos_dict, shoulder_pos_dict)

    if release_frame is None:
        raise ValueError("Could not detect release frame")

    if not hoop_pos_dict:
        raise HTTPException(status_code=422, detail="No hoop detected in video")

    closest_frame = min(
        hoop_pos_dict.keys(),
        key=lambda frame: abs(frame - release_frame)
    )

    hoop_x, hoop_y = hoop_pos_dict[closest_frame][:2]

    person = frame_dict[release_frame].keypoints.xy[0]
    dist = analyze.dist_bw_ball_and_hoop(person, hoop_x) 

    result = frame_dict[release_frame]
    person_box = result.boxes.xyxy[0]
    foot_y = person_box[3].item()
    release_x, release_y = ball_pos_dict[release_frame]
    release_height_pixels = foot_y - release_y
    ground_y = max(
        frame_dict[f].keypoints.xy[0][15:17, 1].max().item()
        for f in frame_dict
        if frame_dict[f].keypoints is not None
        and len(frame_dict[f].keypoints.xy) > 0
    )
    hoop_height_pixels = ground_y - hoop_y
    meters_per_pixel = 3.048 / hoop_height_pixels
    distance_m = dist * meters_per_pixel
    release_height_m = release_height_pixels * meters_per_pixel

    optimal_angle = analyze.optimal_shooting_arc(distance_m, release_height_m)

    angle = analyze.ball_trajectory_arc(release_frame, ball_pos_dict)
    shot_made = analyze.shot_made(release_frame, ball_pos_dict, hoop_pos_dict)

    print("Release frame:", release_frame)
    print("Release position:", (release_x, release_y))
    print("Release angle:", abs(angle))
    print("Distance:", distance_m)
    print("Optimal angle:", optimal_angle)

    return {
        "release_frame": release_frame,
        "release_position": [release_x, release_y],
        "release_angle": abs(angle),
        "distance_meters": float(distance_m),
        "optimal_angle": float(optimal_angle),
        "shot_made": shot_made,
    }

@app.post("/annotated-vid")
async def annotated_vid(vids: UploadFile = File(...), hand: Literal["left", "right"] | None = Form(None)):
    if not vids.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File uploaded is not a video. Got: {vids.content_type}"
        )

    if release_frame is None:
        raise HTTPException(
            status_code=422,
            detail="Analyze a video first to determine the release frame"
        )

    with open("uploaded_annotation_video.mp4", "wb") as buffer:
        shutil.copyfileobj(vids.file, buffer)

    pose_results = analyze.model("uploaded_annotation_video.mp4", stream=True)

    cap = cv2.VideoCapture("uploaded_annotation_video.mp4")

    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Could not create output video")

    pose_iter = iter(pose_results)
    shooting_hand = analyze.ShootingHand[hand.upper()] if hand else analyze.find_shooting_hand(release_frame, ball_pos_dict, wrist_pos_dict)
    elbow_frames = []
    for frame_num in sorted(person_pos_dict):
        if max(0, release_frame - int(fps * 0.5)) <= frame_num <= release_frame - 4:
            angle = analyze.elbow_angle(person_pos_dict[frame_num].cpu().numpy(), shooting_hand)
            if angle is not None and angle > 0:
                elbow_frames.append((frame_num, angle))
    if not elbow_frames:
        cap.release()
        out.release()
        raise HTTPException(status_code=422, detail="Could not measure the elbow before release")
    # Use the last bent-arm position before the arm extends toward release.
    set_frames = [elbow_frames[i] for i in range(1, len(elbow_frames) - 1)
                  if elbow_frames[i][1] <= min(elbow_frames[i-1][1], elbow_frames[i+1][1])]
    elbow_frame, elbow_angle = set_frames[-1] if set_frames else min(elbow_frames, key=lambda item: item[1])
    elbow_text = f"{elbow_angle:.1f} deg" if elbow_angle is not None else "N/A"
    shoulder_idx, elbow_idx, wrist_idx = (6, 8, 10) if shooting_hand == analyze.ShootingHand.RIGHT else (5, 7, 9)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        try:
            result = next(pose_iter)
            print("Keypoints:", result.keypoints.xy.shape if result.keypoints is not None else None)
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                print("Keypoints:", result.keypoints.xy[0].tolist())
        except StopIteration:
            break

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            keypoints = result.keypoints.xy[0].cpu().numpy() 

            for partA, partB in POSE_PAIRS:
                pointA = tuple(map(int, keypoints[partA]))
                pointB = tuple(map(int, keypoints[partB]))

                if pointA != (0, 0) and pointB != (0, 0):
                    cv2.line(frame, pointA, pointB, (0, 255, 0), 4)
                    cv2.circle(frame, pointA, radius=6, color=(0, 0, 255), thickness=-1)
                    cv2.circle(frame, pointB, radius=6, color=(0, 0, 255), thickness=-1)
            shoulder, elbow, wrist = keypoints[[shoulder_idx, elbow_idx, wrist_idx]]
            current_angle = analyze.elbow_angle(keypoints, shooting_hand)
            if current_angle is not None and all(point.any() for point in (shoulder, elbow, wrist)):
                center = tuple(map(int, elbow))
                upper_arm, forearm = shoulder - elbow, wrist - elbow
                start = math.degrees(math.atan2(upper_arm[1], upper_arm[0]))
                end = math.degrees(math.atan2(forearm[1], forearm[0]))
                sweep = (end - start + 180) % 360 - 180
                radius = max(8, int(min(math.hypot(*upper_arm), math.hypot(*forearm)) * 0.3))
                cv2.line(frame, center, tuple(map(int, shoulder)), (0, 255, 255), 4)
                cv2.line(frame, center, tuple(map(int, wrist)), (0, 255, 255), 4)
                cv2.ellipse(frame, center, (radius, radius), 0,
                            min(start, start + sweep), max(start, start + sweep), (0, 255, 255), 4)
                cv2.putText(frame, f"{current_angle:.1f} deg", (center[0] + 30, center[1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"Shooting hand: {shooting_hand.name.title()}",
                    (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(frame, f"Elbow (set, frame {elbow_frame}): {elbow_text}",
                    (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        out.write(frame)

    cap.release()
    out.release()

    return {
        "message": "Annotated video saved", 
        "file": "annotated_video.mp4",
        "shooting_hand": shooting_hand,
        "elbow_angle": elbow_angle,
        "elbow_frame": elbow_frame,
    }

@app.post("/report")
async def analysis_report():
    pass

        
