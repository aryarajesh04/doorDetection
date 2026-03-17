import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "OpenDoor.v7i.yolov8", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")


model = YOLO("yolov8n.yaml")

print(f"Dataset: {DATA_YAML}")
print(f"Output:  {RUNS_DIR}")


results = model.train(
    data=DATA_YAML,
    epochs=1,
    imgsz=416,
    batch=2,           
    device=0,           
    workers=8,
    amp=True,           
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,           
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=20,        
    save=True,
    save_period=10,     
    project=RUNS_DIR,
    name="door_detection",
    pretrained=False,  
    verbose=True,
    plots=True,         # Loss Curves
)

print(f"Best model: {RUNS_DIR}/door_detection/weights/best.pt")
print(f"Metrics:    {RUNS_DIR}/door_detection/results.csv")


print("Test Set")
metrics = model.val(
    data=DATA_YAML,
    split="test",
    device=0,
)

print(f"\nmAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
