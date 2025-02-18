import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import RoIPool

# R-CNNの定義
class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False  # ResNetの重みを固定
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1.0/16)
        self.classifier = nn.Linear(resnet.fc.in_features * 7 * 7, num_classes)
        self.bbox_regressor = nn.Linear(resnet.fc.in_features * 7 * 7, 4)
        
    def forward(self, images, rois):
        features = self.backbone(images)
        pooled_features = self.roi_pool(features, rois)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        class_logits = self.classifier(pooled_features)
        bbox_preds = self.bbox_regressor(pooled_features)
        return class_logits, bbox_preds

class VOCUtils:
    def __init__(self):
        pass

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images, 0), targets

    @staticmethod
    def get_voc_labels(dataset):
        labels = set()
        for _, target in dataset:
            for obj in target['annotation']['object']:
                labels.add(obj['name'])
        sorted_label = sorted(labels)
        return {label: idx for idx, label in enumerate(sorted_label)}