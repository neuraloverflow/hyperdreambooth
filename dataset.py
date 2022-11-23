import random
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset

import torchvision.transforms.functional as TF


class RandomResize:
    def __init__(self, from_size, to_size):
        self.from_size = from_size
        self.to_size = to_size

    def __call__(self, x):
        size = random.randint(self.from_size, self.to_size)
        w, h = x.size
        if w > h:
            h = int(size / w * h)
            w = size
        else:
            w = int(size / h * w)
            h = size
        x = TF.resize(x, (h, w))
        return x


class DreamBoothDatasetWithTags(Dataset):
    """
    A dataset to prepare the instance and class images with the promots for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        tags=None,
        add_pad=True,
        new_word_pairs=[],
    ):
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        instance_data_root = Path(instance_data_root)
        # self.instance_images_path = sorted(list(instance_data_root.glob("**/*.png")))
        self.instance_subdir_path = [d.stem for d in sorted(list(instance_data_root.iterdir()))]
        self.subdir_images_path = [sorted(list((instance_data_root / d).glob("*.png"))) for d in self.instance_subdir_path]
        self.subdir_size_list = [len(d) for d in self.subdir_images_path]
        self.max_size = max(self.subdir_size_list)
        self._length = self.max_size * len(self.instance_subdir_path)

        self.instance_prompt = instance_prompt
        self.tags = tags
        self.add_pad = add_pad
        self.new_word_list = [w.replace("<", "").replace(">", "") for _, w in new_word_pairs]

        self.image_transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                RandomResize(300, 768),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        subdir_index0 = index // self.max_size
        subdir_index1 = (index % self.max_size) % self.subdir_size_list[subdir_index0]
        img_filename = self.subdir_images_path[subdir_index0][subdir_index1]
        instance_image = Image.open(img_filename)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.tags is not None:
            key = f"{img_filename.parent.stem}_{img_filename.stem}"
            if key in self.tags:
                label = img_filename.parent.stem
                info = self.tags[key]
                tag = info["tag"]
                weight = info["weight"]
                property = info["property"]
                if property == "style":
                    header = f"style of {label}"
                else:
                    header = f"{label}"

                header = f"{header} {self.instance_prompt}"

                if isinstance(tag, str):
                    instance_prompt = f"{header}, {tag}"
                elif isinstance(tag, list):
                    tag_list = [key.replace("_", " ").replace("rating:", "") for key, prob in tag if prob > np.random.rand()]
                    random.shuffle(tag_list)
                    tags_str = " ".join(tag_list)
                    instance_prompt = f"{header}, {tags_str}"
            else:
                assert False, key
            example["weight"] = weight
        else:
            instance_prompt = self.instance_prompt
            example["weight"] = 1.0

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="max_length" if self.add_pad else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example
