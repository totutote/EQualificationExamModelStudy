import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from rcnn_util import RCNN, VOCUtils

# 学習の設定
def train_rcnn(model, dataloader_train, dataloader_val, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        # tqdmでdataloaderをラップしてプログレスバーを表示
        with tqdm(total=len(dataloader_train), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for images, targets in dataloader_train:
                images = images.to(device)
                rois = []
                labels = []
                bbox_targets = []
                for target in targets:
                    for object in target['annotation']['object']:
                        bbox = object['bndbox']
                        rois.append([0, int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
                        labels.append(int(label_mapping[object['name']]))
                        bbox_targets.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
                rois = torch.FloatTensor(rois).to(device)
                labels = torch.tensor(labels).to(device)
                bbox_targets = torch.FloatTensor(bbox_targets).to(device)
                
                optimizer.zero_grad()
                class_logits, bbox_preds = model(images, rois)
                loss_cls = criterion(class_logits, labels)
                loss_bbox = criterion(bbox_preds, bbox_targets)
                loss = loss_cls + loss_bbox
                loss.backward()
                optimizer.step()

                # プログレスバーを更新
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
        
        # 検証データでaccuracyを計算
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(dataloader_val), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for images, targets in dataloader_val:
                    images = images.to(device)
                    rois = []
                    labels = []
                    for target in targets:
                        for object in target['annotation']['object']:
                            bbox = object['bndbox']
                            rois.append([0, int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
                            labels.append(int(label_mapping[object['name']]))
                    rois = torch.FloatTensor(rois).to(device)
                    labels = torch.tensor(labels).to(device)
                    
                    class_logits, _ = model(images, rois)
                    _, predicted = torch.max(class_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
                accuracy = 100 * correct / total
                pbar.set_postfix({'accuracy': accuracy})
                pbar.update(1)
            model.train()

# Pascal VOCデータセットの読み込み
transform = Compose([Resize((224, 224)), ToTensor()])
voc_dataset = VOCDetection(root='dataset/', year='2012', image_set='train', download=True, transform=transform)

# データセットをトレーニングと検証に分割
train_size = int(0.8 * len(voc_dataset))
val_size = len(voc_dataset) - train_size
voc_train, voc_val = random_split(voc_dataset, [train_size, val_size])

# データローダーの作成
dataloader_train = DataLoader(voc_train, batch_size=32, num_workers=4, pin_memory=True, collate_fn=VOCUtils.collate_fn)
dataloader_val = DataLoader(voc_val, batch_size=32, num_workers=4, pin_memory=True, collate_fn=VOCUtils.collate_fn)

label_mapping = VOCUtils.get_voc_labels(voc_dataset)

# モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
num_classes = len(label_mapping)
model = RCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # 学習の実行
    train_rcnn(model, dataloader_train, dataloader_val, optimizer, criterion, num_epochs=3)

    # モデルの保存
    torch.save(model.state_dict(), 'rcnn_model.pth')

