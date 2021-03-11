import numpy as np
import torch

from PIL import Image, ImageOps
from torch.utils import data
from torchvision import transforms

from ocr.utils.cafcn_img_generator import CustomTextDataGenerator
from ocr.utils import box_util
from ocr.utils.io import load_json_file, load_image, load_text_source


class CARandomDataset(data.Dataset):
    def __init__(
        self,
        tokenizer,
        text_source,
        font_list,
        train=True,
        text_len=0,
        text_len_warmup=True,
        epoch=0,
        num_iters=5000,
        batch_size=128,
        text_len_warmup_steps=10000,
        small=False
    ):
        self.tokenizer = tokenizer
        self.text_source = text_source
        self.font_list = font_list
        self.text_len = text_len
        self.text_len_warmup = text_len_warmup
        self.length = num_iters * batch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.cur_id = epoch * num_iters * batch_size
        self.max_len = 12
        self.batch_size = batch_size
        self.input_size = (512, 64)
        self.train = train
        self.warmup_steps = text_len_warmup_steps
        self.small = small
        self.char_list = list(tokenizer.dict.keys())[2:]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # random_length = self.get_random_text_len()
        # random_length = np.random.randint(1, 12)
        random_length = np.random.choice([1, 2, 3], p=[0.1, 0.45, 0.45])
        text = self.generate_format_text(random_length)
        options = self.get_random_options(text)
        image, boxes = self.generate_image(index, text, options)
        if options["orientation"] == 1:
            # 垂直文本放平
            boxes = box_util.bbox_rotate90(boxes, image.size)
            image = image.rotate(90, expand=True)

        image, boxes = self.resize_and_padding_image(image, boxes=boxes, with_box=True)
        image = self.transform(image)

        cls_map = self.get_cls_map(text, boxes)
        attn_map2 = self.get_attn_map(boxes, 1 / 4)
        attn_map3 = self.get_attn_map(boxes, 1 / 8)
        attn_map4 = self.get_attn_map(boxes, 1 / 16)
        attn_map5 = self.get_attn_map(boxes, 1 / 32)
        self.cur_id += 1
        results = {
            "image": image,
            "hm": torch.LongTensor(cls_map),
            "a2": torch.FloatTensor(attn_map2),
            "a3": torch.FloatTensor(attn_map3),
            'a4': torch.FloatTensor(attn_map4),
            'a5': torch.FloatTensor(attn_map5),
        }
        if not self.train:
            results['labels'] = [self.tokenizer.encode(x) for x in text]
        return results

    def get_random_text_len(self):
        cur_step = int(self.cur_id / self.batch_size)
        if (
            self.text_len_warmup and cur_step < self.warmup_steps
        ) or self.text_len <= 0:
            text_length = int((cur_step / self.warmup_steps) * self.max_len) + 1
            # text_length = int(self.cur_id / (self.batch_size * warmup_steps))
            text_length = min(text_length, self.max_len)
        elif self.text_len > 0:
            text_length = self.text_len
        else:
            text_length = np.random.randint(1, self.max_len + 1)
        return text_length

    def generate_format_text(self, length):
        if self.small:
            # text = "".join(np.random.choice(self.text_source[50: 80], 1))
            text = "".join(np.random.choice(self.text_source, length))
        else:
            text = "".join(np.random.choice(self.char_list, length))
        return text

    def get_random_options(self, text):
        options = {
            "skewing_angle": 6,
            "random_skew": True,
            "blur": 0,
            "random_blur": False,
            "distorsion_type": 0,
            "distorsion_orientation": 0,
            "is_handwritten": False,
            "alignment": 0,
            "text_color": "#282828",
            "space_width": 1.0,
            "margins": (5, 5, 5, 5),
        }
        if self.small:
            options['font'] = self.font_list[0]
            options["font_size"] = 30
            options["width"] = len(text) * options["font_size"]
            options["background_type"] = 1
            options["orientation"] = 1 if np.random.rand() > 0.5 else 0
        else:
            options['font'] = self.font_list[np.random.randint(0, len(self.font_list))]
            options["font_size"] = np.random.randint(25, 55)
            options["width"] = len(text) * options["font_size"] + np.random.randint(20, 50)
            options["background_type"] = np.random.randint(0, 3)
            options["orientation"] = 1 if np.random.rand() > 0.5 else 0
        # options['fit'] = True if np.random.rand() > 0.5 else False
        return options

    def generate_image(self, index, text, options):
        return CustomTextDataGenerator.generate(
            index,
            text,
            options['font'],
            None,
            options["font_size"],
            None,
            options["skewing_angle"],
            options["random_skew"],
            options["blur"],
            options["random_blur"],
            options["background_type"],
            options["distorsion_type"],
            options["distorsion_orientation"],
            options["is_handwritten"],
            0,
            options["width"],
            options["alignment"],
            options["text_color"],
            options["orientation"],
            options["space_width"],
            margins=options["margins"],
        )
    # @func_line_time
    # @profile

    def resize_and_padding_image(self, image: Image.Image, with_box=False, boxes=None):
        w, h = image.size
        target_w, target_h = self.input_size
        ratio = target_h / float(h)
        new_w, new_h = int(w * ratio), target_h

        if new_w < target_w:
            # add pad
            image = image.resize((new_w, new_h), Image.ANTIALIAS)
            border = 0, 0, target_w - new_w, 0  # left, top, right ,bottom
            image = ImageOps.expand(image, border, fill="white")
            if with_box and boxes is not None:
                boxes = box_util.bbox_resize(boxes, (ratio, ratio))
        else:
            image = image.resize((target_w, target_h), Image.ANTIALIAS)
            if with_box and boxes is not None:
                boxes = box_util.bbox_resize(boxes, (target_w / w, target_h / h))
        if with_box:
            return image, boxes
        return image

    def get_cls_map(self, text, boxes, ratio=0.5):
        map_size = int(self.input_size[1] * ratio), int(self.input_size[0] * ratio)
        cls_map = np.zeros(map_size)
        resized_boxes = box_util.bbox_resize(boxes, (0.5, 0.5))
        for i, box in enumerate(resized_boxes):
            cls_id = self.tokenizer.encode(text[i])
            gt_box = box_util.cal_gt_box(box)
            x_min, y_min, x_max, y_max = gt_box.astype(int)
            cls_map[y_min: y_max + 1, x_min: x_max + 1] = int(cls_id)
        return cls_map

    def get_attn_map(self, boxes, ratio):
        map_size = 1, int(self.input_size[1] * ratio), int(self.input_size[0] * ratio)
        attn_map = np.zeros(map_size)
        resized_boxes = box_util.bbox_resize(boxes, (ratio, ratio))
        for i, box in enumerate(resized_boxes):
            x_min, y_min, x_max, y_max = box
            w, h = x_max - x_min, y_max - y_min
            gt_center = int(x_min + w / 2), int(y_min + h / 2)
            gt_w, gt_h = int(max(2, w * 0.5)), int(max(2, h * 0.5))

            xg_min, yg_min, xg_max, yg_max = int(gt_center[0] - gt_w / 2), int(gt_center[1] - gt_h / 2), int(gt_center[0] + gt_w / 2), int(gt_center[1] + gt_h / 2)
            attn_map[0][yg_min: yg_max + 1, xg_min: xg_max + 1] = 1
        return attn_map


class CASyntheticDataset(CARandomDataset):
    def __init__(
        self,
        tokenizer,
        text_source,
        data_dir,
        img_w,
        img_h,
        train=True,
    ):
        self.tokenizer = tokenizer
        self.text_source = load_text_source(text_source)
        self.data_dir = data_dir
        
        self.input_size = (img_w, img_h)
        self.train = train
        self.transform = transforms.Compose(
            [
                # transforms.GaussianBlur(1),
                transforms.ToTensor(),
                # transforms.Normalize(0.5, 0.5)
            ]
        )

    def __len__(self):
        return len(self.text_source)

    def __getitem__(self, index):
        image_id = self.text_source[index]
        image_path = f'{self.data_dir}/imgs/{image_id}.jpg'
        label_path = f'{self.data_dir}/labels/{image_id}.json'
        info = load_json_file(label_path)
        text = info['text']
        boxes = info['bboxes']
        image = load_image(image_path)
        w, h = image.size
        if w < h:
            # 垂直文本放平
            boxes = box_util.bbox_rotate90(boxes, image.size)
            image = image.rotate(90, expand=True)

        image, boxes = self.resize_and_padding_image(image, boxes=boxes, with_box=True)
        image = self.transform(image)

        cls_map = self.get_cls_map(text, boxes)
        attn_map2 = self.get_attn_map(boxes, 1 / 4)
        attn_map3 = self.get_attn_map(boxes, 1 / 8)
        attn_map4 = self.get_attn_map(boxes, 1 / 16)
        attn_map5 = self.get_attn_map(boxes, 1 / 32)
        targets = {
            "image": image,
            "hm": torch.LongTensor(cls_map),
            "a2": torch.FloatTensor(attn_map2),
            "a3": torch.FloatTensor(attn_map3),
            'a4': torch.FloatTensor(attn_map4),
            'a5': torch.FloatTensor(attn_map5),
        }
        if not self.train:
            targets['labels'] = [self.tokenizer.encode(x) for x in text]
        return image, targets
