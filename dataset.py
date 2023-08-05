import os
from tqdm import tqdm
from datasets import load_dataset

def create_dataset(data, split):
  data = data[split]
  print(f'Running for {split} split...')
  for idx, sample in tqdm(enumerate(data), total=len(data)):
    image = sample['image']
    labels = sample['litter']['label']
    bboxes = sample['litter']['bbox']
    targets = []
    # creating the label txt files
    for label, bbox in zip(labels, bboxes):
      targets.append(f'{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}')
    with open(f'datasets/labels/{split}/{idx}.txt', 'w') as f:
      for target in targets:
        f.write(target+'\n')
    image.save(f'datasets/images/{split}/{idx}.png')

if __name__ == '__main__':
    dataset = load_dataset('Kili/plastic_in_river', num_proc=12)

    os.makedirs('datasets/images/train', exist_ok=True)
    os.makedirs('datasets/images/validation', exist_ok=True)
    os.makedirs('datasets/labels/train', exist_ok=True)
    os.makedirs('datasets/labels/validation', exist_ok=True)

    create_dataset(dataset, 'train')
    create_dataset(dataset, 'validation')