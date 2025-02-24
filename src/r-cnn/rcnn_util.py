import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import RoIPool
from torchvision.ops import box_iou

# R-CNNの定義
class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.roi_pool = RoIPool(output_size=(4, 4), spatial_scale=1.0/32)
        self.classifier = nn.Linear(resnet.fc.in_features * 4 * 4, num_classes)
        self.bbox_regressor = nn.Linear(resnet.fc.in_features * 4 * 4, 4)
        
    def forward(self, images, rois):
        features = self.backbone(images)
        pooled_features = self.roi_pool(features, rois)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        class_logits = self.classifier(pooled_features)
        bbox_preds = self.bbox_regressor(pooled_features)
        return class_logits, bbox_preds

class VOCUtils:

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        images = torch.stack(images)
        return images, targets

    @staticmethod
    def get_voc_labels(voc_dataset):
        labels = set()
        for i in range(len(voc_dataset)):
            annotation = voc_dataset[i][1]
            for object in annotation['annotation']['object']:
                labels.add(object['name'])
        labels = sorted(list(labels))
        label_mapping = {label: i for i, label in enumerate(labels)}
        return label_mapping

    @staticmethod
    def extract_roi_data(targets, label_mapping, device, resized_width=448, resized_height=448):
        rois = []
        labels = []
        bbox_targets = []
        for batch_index, target in enumerate(targets):
            original_width = int(target['annotation']['size']['width'])
            original_height = int(target['annotation']['size']['height'])
            width_ratio = resized_width / original_width
            height_ratio = resized_height / original_height

            for object in target['annotation']['object']:
                bbox = object['bndbox']
                x_min = int(bbox['xmin'])
                y_min = int(bbox['ymin'])
                x_max = int(bbox['xmax'])
                y_max = int(bbox['ymax'])

                x_min_resized = int(x_min * width_ratio)
                y_min_resized = int(y_min * height_ratio)
                x_max_resized = int(x_max * width_ratio)
                y_max_resized = int(y_max * height_ratio)

                rois.append([batch_index, x_min_resized, y_min_resized, x_max_resized, y_max_resized])
                labels.append(int(label_mapping[object['name']]))
                bbox_targets.append([x_min_resized, y_min_resized, x_max_resized, y_max_resized])

        rois = torch.FloatTensor(rois).to(device)
        labels = torch.tensor(labels).to(device)
        bbox_targets = torch.FloatTensor(bbox_targets).to(device)
        return rois, labels, bbox_targets
    
    @staticmethod
    def calculate_iou(bbox_preds, bbox_targets):
        # bbox_predsとbbox_targetsを[N, 4]の形式に変換
        bbox_preds = bbox_preds.view(-1, 4)
        bbox_targets = bbox_targets.view(-1, 4)
        
        # IoUを計算
        iou = box_iou(bbox_preds, bbox_targets)
        
        # 各予測バウンディングボックスに対する最大のIoUを返す
        max_iou_per_bbox = torch.max(iou, dim=1)[0]
        
        return max_iou_per_bbox