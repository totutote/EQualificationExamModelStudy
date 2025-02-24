import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader, Subset
from rcnn_util import RCNN, VOCUtils

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 推論の設定
def inference_rcnn(model, images, rois):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        rois = rois.to(device)
        class_logits, bbox_preds = model(images, rois)
        return class_logits, bbox_preds

# Pascal VOCデータセットの読み込み
transform = Compose([Resize((224, 224)), ToTensor()])
voc_train = VOCDetection(root='dataset/', year='2012', image_set='train', download=False, transform=transform)

# データセットのサブセットを作成（例: 最初の100サンプルのみ）
subset_indices = list(range(100))
voc_subset = Subset(voc_train, subset_indices)

# DataLoaderの設定
dataloader = DataLoader(voc_subset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_fn)

label_mapping = VOCUtils.get_voc_labels(voc_train)

if __name__ == '__main__':
    # モデルの読み込み
    model = RCNN(num_classes=len(label_mapping)).to(device)
    model.load_state_dict(torch.load('rcnn_model.pth', weights_only=True))

    # 推論の実行
    for i, (images, targets) in enumerate(dataloader):
        if i >= 3:
            break
        rois = []
        for target in targets:
            for obj in target['annotation']['object']:
                bbox = obj['bndbox']
                rois.append([0, int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
        rois = torch.FloatTensor(rois).to(device)
        class_logits, bbox_preds = inference_rcnn(model, images, rois)
        print(f"Sample {i+1}:")
        print("Class logits:", class_logits)
        print("Bounding box predictions:", bbox_preds)