import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCDetection
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import transforms

RESIZE_SIZE = 300  # リサイズ後の画像サイズ

LABEL_MAP = {
    "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
    "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
    "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
    "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20,
}

class NormalizeAnnotations:
    def __call__(self, image, target):
        width, height = RESIZE_SIZE, RESIZE_SIZE
        orig_width = float(target["annotation"]["size"]["width"])
        orig_height = float(target["annotation"]["size"]["height"])
        for obj in target["annotation"]["object"]:
            bndbox = obj["bndbox"]
            # リサイズ後のバウンディングボックスの座標を計算し正規化
            bndbox["xmin"] = (float(bndbox["xmin"]) * (width / orig_width)) / width
            bndbox["ymin"] = (float(bndbox["ymin"]) * (height / orig_height)) / height
            bndbox["xmax"] = (float(bndbox["xmax"]) * (width / orig_width)) / width
            bndbox["ymax"] = (float(bndbox["ymax"]) * (height / orig_height)) / height
        return image, target


class AdjustOriginBBox:
    def __call__(self, image, target):
        for obj in target["annotation"]["object"]:
            bndbox = obj["bndbox"]
            # VOCアノテーションの原点は1,1なので0,0に調整
            bndbox["xmin"] = float(bndbox["xmin"]) - 1
            bndbox["ymin"] = float(bndbox["ymin"]) - 1
            bndbox["xmax"] = float(bndbox["xmax"]) - 1
            bndbox["ymax"] = float(bndbox["ymax"]) - 1
        return image, target


class PrintData:
    def __call__(self, image, target):
        print("Data before any transform:", image.size)
        print("Target before any transform:", target)
        return image, target


class ResizeWithTarget:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 画像のリサイズ
        image = F.resize(image, self.size, antialias=True)
        new_width, new_height = F.get_image_size(image)

        # 元の画像サイズをXMLから取得
        orig_width = float(target["annotation"]["size"]["width"])
        orig_height = float(target["annotation"]["size"]["height"])
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # 各オブジェクトのバウンディングボックスをリサイズ
        for obj in target["annotation"]["object"]:
            bndbox = obj["bndbox"]
            bndbox["xmin"] = str(float(bndbox["xmin"]) * scale_x)
            bndbox["ymin"] = str(float(bndbox["ymin"]) * scale_y)
            bndbox["xmax"] = str(float(bndbox["xmax"]) * scale_x)
            bndbox["ymax"] = str(float(bndbox["ymax"]) * scale_y)

        # サイズ情報を更新
        target["annotation"]["size"]["width"] = str(new_width)
        target["annotation"]["size"]["height"] = str(new_height)

        return image, target


class ToTensorWithTarget:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if isinstance(target, dict):
            for key in ["boxes", "labels", "image_id", "area", "iscrowd"]:
                if key in target and not torch.is_tensor(target[key]):
                    target[key] = torch.tensor(target[key])
        return image, target


class MyCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def multiobject_collate_fn(batch):
    imgs = []
    boxes = []
    labels = []
    image_ids = []
    areas = []
    iscrowds = []

    for idx, (image, target) in enumerate(batch):
        imgs.append(image)
        objects = target["annotation"]["object"]
        # オブジェクトが単体の場合はリスト化
        if isinstance(objects, dict):
            objects = [objects]
        boxes_list = []
        labels_list = []
        area_list = []
        iscrowd_list = []
        for obj in objects:
            bndbox = obj["bndbox"]
            xmin = float(bndbox["xmin"])
            ymin = float(bndbox["ymin"])
            xmax = float(bndbox["xmax"])
            ymax = float(bndbox["ymax"])
            label = LABEL_MAP.get(obj["name"], 0)
            boxes_list.append([xmin, ymin, xmax, ymax])
            labels_list.append(label)
            # areaを計算
            area = (xmax - xmin) * (ymax - ymin)
            area_list.append(area)
            # iscrowdはデフォルトで0とする
            iscrowd_list.append(0)

        boxes.append(torch.tensor(boxes_list, dtype=torch.float32))
        labels.append(torch.tensor(labels_list, dtype=torch.int64))
        image_ids.append(torch.tensor([idx], dtype=torch.int64))  # バッチ内のインデックスをIDとする
        areas.append(torch.tensor(area_list, dtype=torch.float32))
        iscrowds.append(torch.tensor(iscrowd_list, dtype=torch.int64))

    imgs = torch.stack(imgs, dim=0)
    
    # targetsの形式に合わせて辞書を作成
    targets = []
    for i in range(len(boxes)):
        target = {
            "boxes": boxes[i],
            "labels": labels[i],
            "image_id": image_ids[i],
            "area": areas[i],
            "iscrowd": iscrowds[i]
        }
        targets.append(target)
    
    return imgs, targets


if __name__ == "__main__":
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    transforms = MyCompose(
        [
            # PrintData(),
            ResizeWithTarget((RESIZE_SIZE, RESIZE_SIZE)),
            AdjustOriginBBox(),
            ToTensorWithTarget(),
            NormalizeAnnotations(),
        ]
    )

    # データセットとDataLoaderの準備
    voc_dataset = VOCDetection(
        root="dataset/",
        year="2012",
        image_set="train",
        download=True,
        transforms=transforms,
    )

    dataloader = DataLoader(
        dataset=voc_dataset, batch_size=4, collate_fn=multiobject_collate_fn
    )

    # DataLoaderからバッチを取得
    '''
    images, targets = next(iter(dataloader))

    print("Batch Original Image Size:", images[0].size())
    print("Batch Original Target:", targets[0])
    '''

    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        # トレーニングモードを明示的に設定
        model.train()
        
        # tqdmを使用してプログレスバーを表示
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 画像とターゲットをGPUに移動
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 勾配をリセット
            optimizer.zero_grad()

            # モデルに入力して損失を計算
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 合計損失で逆伝播とパラメータ更新
            losses.backward()
            optimizer.step()

            # 個々の損失をログ出力
            bbox_loss = loss_dict['bbox_regression'].item()
            cls_loss = loss_dict['classification'].item()
            total_loss = losses.item()
            
            print(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}, Bbox: {bbox_loss:.4f}, Cls: {cls_loss:.4f}")

        # モデルの保存
        torch.save(model.state_dict(), f"ssd300_vgg16_epoch_{epoch+1}.pth")