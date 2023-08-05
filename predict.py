from ultralytics import YOLO
from PIL import Image

img = dataset['test'][20]['image']

#choosing the best training checkpoint
model = YOLO('runs/detect/train/weights/best.pt')

pred = model.predict(img)[0]
print(pred.boxes)

#plotting the image with bounding boxes
pred = pred.plot(line_width=1)
#convert from BGR to RGB
pred_rgb = pred[..., ::-1]
pred_img = Image.fromarray(pred_rgb)

pred_img.save('output.png')