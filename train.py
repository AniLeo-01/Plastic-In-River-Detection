from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data = 'plastic.yaml',
    epochs = 20,
    imgsz = (1280, 720), #(w, h)
    batch = 16,
    optimizer = 'Adam',
    lr0 = 1e-3,
    resume=True
)