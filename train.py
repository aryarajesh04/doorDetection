import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "OpenDoor.v7i.yolov8", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")


def main():
    model = YOLO("yolov8s.yaml")

    print(f"Dataset: {DATA_YAML}")
    print(f"Output:  {RUNS_DIR}")

    results = model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        amp=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        patience=30,
        mosaic=1.0,
        mixup=0.1,
        label_smoothing=0.1,
        save=True,
        save_period=10,
        project=RUNS_DIR,
        name="door_detection",
        verbose=True,
        plots=True,
    )

    best_model_path = os.path.join(RUNS_DIR, "door_detection", "weights", "best.pt")
    results_csv_path = os.path.join(RUNS_DIR, "door_detection", "results.csv")

    print(f"Best model: {best_model_path}")
    print(f"Metrics:    {results_csv_path}")

    print("Test Set")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=0,
        workers=8
    )

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()