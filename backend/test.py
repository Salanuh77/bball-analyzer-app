
import requests

video_path = "single_three_point.mp4"
base_url = "http://127.0.0.1:8080"

with open(video_path, "rb") as video:
    print("Sending video for analysis...")

    response = requests.post(
        f"{base_url}/videos",
        files={"vids": ("single_three_point.mp4", video, "video/mp4")},
        timeout=120
    )

print("Analysis status:", response.status_code)
print("Analysis response:", response.text)


with open(video_path, "rb") as video:
    print("Sending video for annotation...")

    response2 = requests.post(
        f"{base_url}/annotated-vid",
        data={"hand": "right"},
        files={"vids": ("single_three_point.mp4", video, "video/mp4")},
        timeout=120
    )

print("Annotation status:", response2.status_code)

if response2.status_code == 200:
    if "video/mp4" in response2.headers.get("content-type", ""):
        with open("annotated_result.mp4", "wb") as f:
            f.write(response2.content)

        print("Saved annotated_result.mp4")
    else:
        print("Annotation response:", response2.text)
        print("Check backend/annotated_video.mp4")
else:
    print("Annotation error:", response2.text)
