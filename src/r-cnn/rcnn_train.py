import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from rcnn_util import RCNN, VOCUtils

def train_rcnn(model, dataloader_train, dataloader_val, optimizer, criterion_cls, criterion_bbox, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        # tqdmでdataloaderをラップしてプログレスバーを表示
        with tqdm(total=len(dataloader_train), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for images, targets in dataloader_train:
                images = images.to(device)
                rois, labels, bbox_targets = VOCUtils.extract_roi_data(targets, label_mapping, device)
                
                optimizer.zero_grad()
                class_logits, bbox_preds = model(images, rois)
                loss_cls = criterion_cls(class_logits, labels)
                loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
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
        total_iou = 0.0  # IoUの合計を初期化
        num_samples = 0  # サンプル数を初期化
        with torch.no_grad():
            with tqdm(total=len(dataloader_val), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for images, targets in dataloader_val:
                    images = images.to(device)
                    rois, labels, bbox_targets = VOCUtils.extract_roi_data(targets, label_mapping, device)

                    class_logits, bbox_preds = model(images, rois)
                    _, predicted = torch.max(class_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # IoUを計算
                    iou = VOCUtils.calculate_iou(bbox_preds, bbox_targets)
                    total_iou += iou.sum().item()  # バッチ内のIoUの合計を加算
                    num_samples += len(iou)  # サンプル数を加算
            
                accuracy = 100 * correct / total
                mean_iou = total_iou / num_samples  # 平均IoUを計算
                pbar.set_postfix({'accuracy': accuracy, 'mean_iou': mean_iou})  # 平均IoUをプログレスバーに表示
                pbar.update(1)
            model.train()


if __name__ == '__main__':
    start_time_1 = time.time()

    # Pascal VOCデータセットの読み込み
    transform = Compose([Resize((448, 448)), ToTensor()])
    voc_dataset = VOCDetection(root='dataset/', year='2012', image_set='train', download=True, transform=transform)

    # データセットをトレーニングと検証に分割
    train_size = int(0.8 * len(voc_dataset))
    val_size = len(voc_dataset) - train_size
    voc_train, voc_val = random_split(voc_dataset, [train_size, val_size])

    # データローダーの作成
    dataloader_train = DataLoader(voc_train, batch_size=16, num_workers=4, pin_memory=True, collate_fn=VOCUtils.collate_fn)
    dataloader_val = DataLoader(voc_val, batch_size=16, num_workers=4, pin_memory=True, collate_fn=VOCUtils.collate_fn)

    label_mapping = VOCUtils.get_voc_labels(voc_dataset)

    # モデルの初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = len(label_mapping)
    model = RCNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    end_time_1 = time.time()

    print(f"処理1にかかった時間: {end_time_1 - start_time_1:.4f}秒")

    # 学習の実行
    train_rcnn(model, dataloader_train, dataloader_val, optimizer, criterion_cls, criterion_bbox, num_epochs=5)

    # モデルの保存
    torch.save(model.state_dict(), 'rcnn_model.pth')

