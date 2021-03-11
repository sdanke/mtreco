if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'F:\\workspace\\code\\projects\\mtreco')

from torch.utils.data import Dataset
from ocr.utils.io import load_image
from ocr.utils.img_util import rotate_resize_padding_image
from tqdm import tqdm
from torchvision import transforms


class SyntheticDataset(Dataset):
    def __init__(self, data_path, input_size=(192, 48), base_dir='F:/workspace/dataset/synthetic_line_images'):
        super(SyntheticDataset, self).__init__()
        self.input_size = input_size
        self.totensor = transforms.ToTensor()
        self.base_dir = base_dir
        self.imgs, self.labels = self.load_data(data_path)

    def load_data(self, data_path):
        imgs = []
        labels = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Loading dataset'):
                img_id, label = line.replace('\n', '').split('[SEP]')
                img_path = f'{self.base_dir}/imgs/{img_id}.jpg'
                imgs.append(img_path)
                labels.append(label)
        return imgs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]

        img = load_image(img_path)
        img = rotate_resize_padding_image(img, self.input_size)
        img = self.totensor(img).sub_(0.5).div_(0.5)
        return img, label

    # def get_batch_data(self):
    #     balanced_batch_images = []
    #     balanced_batch_texts = []

    #     for i, data_loader_iter in enumerate(self.dataloader_iter_list):
    #         try:
    #             image, text = data_loader_iter.next()
    #             balanced_batch_images.append(image)
    #             balanced_batch_texts += text
    #         except StopIteration:
    #             self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
    #             image, text = self.dataloader_iter_list[i].next()
    #             balanced_batch_images.append(image)
    #             balanced_batch_texts += text
    #         except ValueError:
    #             pass

    #     balanced_batch_images = torch.cat(balanced_batch_images, 0)

    #     return balanced_batch_images, balanced_batch_texts


if __name__ == '__main__':
    dataset = SyntheticDataset('F:\\workspace\\code\\projects\\mtreco\\data\\train.txt')
    a = dataset[0]
    print(len(a))