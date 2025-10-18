import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ListImageDataset(Dataset):
    def __init__(
        self, list_txt, root=None, img_size=224, is_train=True, label_to_index=None
    ):
        self.samples = []
        list_dir = os.path.dirname(os.path.abspath(list_txt))

        # Read and parse the list file robustly
        with open(list_txt, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip().replace("\r", "")
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]

                # Normal case: "path,label"
                if len(parts) == 2:
                    path, label = parts[0], parts[1]

                # Common broken case: "basename,ext,label" e.g. "JPCLN002,png,female"
                elif len(parts) == 3 and parts[1].lower() in (
                    "png",
                    "jpg",
                    "jpeg",
                    "bmp",
                    "tif",
                    "tiff",
                    "webp",
                ):
                    path = parts[0] + "." + parts[1]
                    label = parts[2]

                # If there are more commas but the last one separates label: use rsplit
                else:
                    try:
                        path, label = line.rsplit(",", 1)
                        path = path.strip()
                        label = label.strip()
                    except ValueError:
                        raise ValueError(
                            f"Bad line {lineno} in {list_txt}: '{line}' (expected 'path,label' or 'path,ext,label')"
                        )

                path = path.replace("\\", "/")

                # Try resolving the path from a few candidate locations
                candidates = [path, os.path.join(list_dir, path)]
                if root:
                    candidates.append(os.path.join(root, path))
                    # also try root + basename in case path included directories that shouldn't be there
                    candidates.append(os.path.join(root, os.path.basename(path)))

                resolved = None
                for c in candidates:
                    c_norm = os.path.normpath(c)
                    if os.path.exists(c_norm):
                        resolved = c_norm
                        break

                if resolved is None:
                    raise FileNotFoundError(
                        f"Image file not found for line {lineno} in {list_txt}: tried {candidates}"
                    )

                self.samples.append((resolved, label))

        # Build or accept label->index mapping
        if label_to_index is None:
            uniq = sorted({lab for _, lab in self.samples})
            self.label_to_index = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label_to_index = label_to_index

        # Transforms (RGB, normalized for 3 channels)
        if is_train:
            tf_list = [
                transforms.Resize((img_size, img_size)),
                # insert augmentations here if desired, e.g. RandomHorizontalFlip()
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        else:
            tf_list = [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        self.tf = transforms.Compose(tf_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        try:
            # Open as RGB (3 channels) to match normalization above
            with Image.open(path) as img:
                img = img.convert("RGB")
                x = self.tf(img)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            raise RuntimeError(f"Failed to open image '{path}': {e}")

        if label_str not in self.label_to_index:
            raise KeyError(f"Label '{label_str}' not in label_to_index mapping")

        y = self.label_to_index[label_str]
        return x, torch.tensor(y, dtype=torch.long)


def make_loaders(
    train_txt, test_txt, root=None, img_size=224, batch_size=128, num_workers=4, pin_memory=True
):
    train_ds = ListImageDataset(train_txt, root, img_size, is_train=True)
    test_ds = ListImageDataset(
        test_txt, root, img_size, is_train=False, label_to_index=train_ds.label_to_index
    )

    # Note: on Windows, be careful with num_workers>0; set to 0 if you get Windows fork errors.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, test_loader, len(train_ds.label_to_index)
