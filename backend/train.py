from ultralytics import YOLO

hoop_model = YOLO("yolov8n.pt")

hoop_model.train(
    data="data.yaml",
    epochs=30
)

