import requests

print("Starting test...")

url = "http://127.0.0.1:8080/videos"

with open("single_three_point.mp4", "rb") as video:
    print("Sending video...")

    response = requests.post(
        url,
        files={"vids": ("single_three_point.mp4", video, "video/mp4")},
        timeout=120
    )

print("Status:", response.status_code)
print("Response:", response.text)