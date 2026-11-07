from fastapi import FastAPI, UploadFile, File, HTTPException, status
import shutil
from pydantic import BaseModel, Field
import analyze
import math
import asyncio

app = FastAPI()

class User(BaseModel):
    id: int
    username: str
    curVid: str
    vids: list[str]

@app.post("/videos") 
async def analyze_videos(vids: UploadFile = File(...)):
    if not vids.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File uploaded is not a video. Got: {vids.content_type}")
    
    with open("uploaded_video.mp4", "wb") as buffer:
        shutil.copyfileobj(vids.file, buffer)

    pose_results = analyze.model("uploaded_video.mp4", stream=True)
    hoop_results = analyze.hoop_model("uploaded_video.mp4",
        stream=True,
        imgsz=1280,
        conf=0.05
    )
    ball_results = analyze.ball_model("uploaded_video.mp4", stream=True)
    person_pos_dict = {}
    hoop_pos_dict = {}


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
                hoop_pos_dict[frame_num] = analyze.hoop_center(box)

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

    hoop_x, hoop_y = hoop_pos_dict[closest_frame]

    person = frame_dict[release_frame].keypoints.xy[0]
    dist = analyze.dist_bw_ball_and_hoop(person, hoop_x, hoop_y) 

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
    release_height_m = release_height_pixels * meters_per_pixel

    optimal_angle = analyze.optimal_shooting_arc(dist, release_height_m)

    angle = analyze.ball_trajectory_arc(release_frame, ball_pos_dict)

    print("Release frame:", release_frame)
    print("Release position:", (release_x, release_y))
    print("Release angle:", abs(angle))
    print("Distance:", dist)
    print("Optimal angle:", optimal_angle)

    return {
        "release_frame": release_frame,
        "release_position": [release_x, release_y],
        "release_angle": abs(angle),
        "distance_pixels": float(dist),
        "optimal_angle": float(optimal_angle),
    }

@app.post("/report")
async def analysis_report():
    pass

        
