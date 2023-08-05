from ultralytics import YOLO
from PIL import Image
from datasets import load_dataset

def predict(image: Image, model_path: str):
    model = YOLO(model_path)
    pred = model.predict(image)[0]
    # print(pred.boxes)
    #plotting the image with bounding boxes
    pred = pred.plot(line_width=1)
    #convert from BGR to RGB
    pred_rgb = pred[..., ::-1]
    pred_img = Image.fromarray(pred_rgb)
    return pred_img

if __name__ == "__main__":
    dataset = load_dataset('Kili/plastic_in_river', num_proc=12)
    id = 20
    img = dataset['test'][id]['image']
    #choosing the best training checkpoint
    model_path = 'runs/detect/train/weights/best.pt'
    pred_img = predict(image=img, model_path=model_path)
    pred_img.save('output.png')

