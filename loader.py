import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ListImageDataset(Dataset):
    def __init__(
        self, list_txt, root=None, img_size=224, is_train=True, label_to_index=None
    ):
        self.samples = []

        list_dir = os.path.dirname(os.path.abspath(list_txt))

        with open(list_txt, "r") as f:
            for line in f:
                line = line.strip().replace("\r", "")
                if not line:
                    continue
                path, label = line.split(",")
                path = path.replace("\\", "/")
                if not os.path.isabs(path) and not os.path.exists(path):
                    if root:
                        path = os.path.join(root, path)
                    else:
                        path = os.path.join(list_dir, path)

                path = os.path.normpath(path)
                self.samples.append((path, label))

        if label_to_index is None:
            uniq = sorted({lab for _, lab in self.samples})
            self.label_to_index = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label_to_index = label_to_index

        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        img = Image.open(path).convert("L")
        x = self.tf(img)
        y = self.label_to_index[label_str]
        return x, y


def make_loaders(
    train_txt, test_txt, root=None, img_size=224, batch_size=128, num_workers=4
):
    train_ds = ListImageDataset(train_txt, root, img_size, is_train=True)
    test_ds = ListImageDataset(
        test_txt, root, img_size, is_train=False, label_to_index=train_ds.label_to_index
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, len(train_ds.label_to_index)
