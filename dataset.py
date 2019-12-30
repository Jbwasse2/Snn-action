import torchvision.datasets as datasets
from pathlib import Path
import torch
import torchvision.transforms.functional as F
import glob
import os
from pathlib import Path

#I made some changes to this dataset, and even made a PR to the pytorch vision repo.
#Until the changes get accepted (or if they get) denied I will put my changes here
class ucf101(datasets.UCF101):
    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = []
        for i in range(len(video_list)):
            path = Path(video_list[i])
            if str(path.relative_to(path.parent.parent)) in selected_files:
                indices.append(i)
        return indices
    def __getitem__(self, idx):
        try:
            video, audio, info, video_idx = self.video_clips.get_clip(idx)
            label = self.samples[self.indices[video_idx]][1]

            if self.transform is not None:
                transformed_video = []
                for counter, image in enumerate(video):
                    image = self.transform(image)
                    transformed_video.append(image)
                video = torch.stack(transformed_video)

            return video, label
        except Exception as e:
            video, audio, info, video_idx = self.video_clips.get_clip(0)
            label = self.samples[self.indices[video_idx]][1]

            if self.transform is not None:
                transformed_video = []
                for counter, image in enumerate(video):
                    image = self.transform(image)
                    transformed_video.append(image)
                video = torch.stack(transformed_video)

            return video, label
