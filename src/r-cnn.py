import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.ops import RoIPool

# R-CNNの定義
class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 最後の全結合層を除く
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1.0/16)
        self.classifier = nn.Linear(2048 * 7 * 7, num_classes)
        self.bbox_regressor = nn.Linear(2048 * 7 * 7, 4)
        
    def forward(self, images, rois):
        features = self.backbone(images)
        pooled_features = self.roi_pool(features, rois)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        class_logits = self.classifier(pooled_features)
        bbox_preds = self.bbox_regressor(pooled_features)
        return class_logits, bbox_preds

# 学習の設定
def train_rcnn(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, rois, labels, bbox_targets in dataloader:
            optimizer.zero_grad()
            class_logits, bbox_preds = model(images, rois)
            loss_cls = criterion(class_logits, labels)
            loss_bbox = criterion(bbox_preds, bbox_targets)
            loss = loss_cls + loss_bbox
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 推論の設定
def inference_rcnn(model, images, rois):
    model.eval()
    with torch.no_grad():
        class_logits, bbox_preds = model(images, rois)
        return class_logits, bbox_preds

# モデルの初期化
num_classes = 21  # クラス数（背景を含む）
model = RCNN(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ダミーデータローダー（実際のデータローダーを使用する必要があります）
dataloader = [(
    torch.randn(1, 3, 224, 224),  # 画像
    torch.FloatTensor([[0, 0, 0, 50, 50]]),  # RoI (batch_index, x1, y1, x2, y2)
    torch.tensor([1]),  # ラベル
    torch.FloatTensor([[0, 0, 50, 50]])  # バウンディングボックスターゲット
)]

# 学習の実行
train_rcnn(model, dataloader, optimizer, criterion, num_epochs=10)

# 推論の実行
images = torch.randn(1, 3, 224, 224)
rois = torch.FloatTensor([[0, 0, 0, 50, 50]])  # RoI (batch_index, x1, y1, x2, y2)
class_logits, bbox_preds = inference_rcnn(model, images, rois)
print(class_logits, bbox_preds)