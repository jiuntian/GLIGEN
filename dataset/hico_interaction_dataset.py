import glob
import json
import os
import random

import torch

from dataset.base_dataset import BaseDataset, recalculate_box_and_verify_if_valid
from .tsv_dataset import project


class HICOInteractionDataset(BaseDataset):
    def __init__(self,
                 dataset_path,
                 which_layer_text='before',
                 which_layer_image="after_reproject",
                 prob_use_caption=1,
                 # random_drop_embedding='none',
                 image_size=512,
                 min_box_size=0.001,
                 max_boxes_per_data=8,
                 max_images=None,  # set as 30K used to eval
                 random_crop=False,
                 random_flip=True,
                 ):
        super().__init__(random_crop, random_flip, image_size)
        self.dataset_path = dataset_path
        self.which_layer_text = which_layer_text
        self.which_layer_image = which_layer_image
        self.prob_use_caption = prob_use_caption
        # self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        print(f"min_box_size {min_box_size}")
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images

        assert which_layer_text in ['before', 'after']
        assert which_layer_image in ['after', 'after_renorm', 'after_reproject']
        # assert random_drop_embedding in ['none', 'both', 'image']

        # Last linear layer used in CLIP text encoder.
        # Here we use it to map CLIP image embedding into penultimate text space. See Appendix in paper.
        self.projection_matrix = torch.load('projection_matrix')

        # Load tsv data
        self.files = glob.glob(os.path.join(self.dataset_path, 'embed_*.clip.pt'))
        assert len(self.files) > 0, 'No file found!'

        # preprocessed CLIP feature embedding length: 768
        self.embedding_len = 768

    def total_images(self):
        return len(self)

    def get_item(self, index):
        item = torch.load(self.files[index], map_location="cpu")
        return item

    def mapping(self, image_embedding):
        if self.which_layer_image == 'after':
            # use CLIP image feaure, the aligned feature space with norm=1.
            return image_embedding
        elif self.which_layer_image == 'after_renorm':
            # same as before but normalize it to 28.7, which is empirically same as text penultimate feature norm.
            return image_embedding * 28.7
        elif self.which_layer_image == 'after_reproject':
            # Re-project the CLIP image feature into text penultimate space using text linear matrix and norm it into 28.7
            image_embedding = project(image_embedding.unsqueeze(0), self.projection_matrix.T)
            image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding / image_embedding.norm()
            image_embedding = image_embedding * 28.7
            return image_embedding

    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        raw_item = self.get_item(index)

        out = {}

        # -------------------- id and image ------------------- #
        out['id'] = raw_item['data_id']
        image = raw_item['image']
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor

        # -------------------- grounding token ------------------- #
        annos = raw_item["hois"]

        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_image_embeddings = []

        text_embedding_name = 'text_embedding_before' if self.which_layer_text == 'before' else 'text_embedding_after'
        image_embedding_name = 'image_embedding_after'

        for anno in annos:
            s_x, s_y, s_w, s_h = anno['subject_xywh']
            s_valid, (s_x0, s_y0, s_x1, s_y1) = recalculate_box_and_verify_if_valid(s_x, s_y, s_w, s_h, trans_info,
                                                                                  self.image_size, self.min_box_size)
            o_x, o_y, o_w, o_h = anno['object_xywh']
            o_valid, (o_x0, o_y0, o_x1, o_y1) = recalculate_box_and_verify_if_valid(o_x, o_y, o_w, o_h, trans_info,
                                                                                  self.image_size, self.min_box_size)
            if s_valid and o_valid:
                areas.append((s_x1 - s_x0) * (s_y1 - s_y0) + (o_x1 - o_x0) * (o_y1 - o_y0))  # area = subject + object
                all_boxes.append(torch.tensor([s_x0, s_y0, s_x1, s_y1]) / self.image_size)  # scale to 0-1
                all_boxes.append(torch.tensor([o_x0, o_y0, o_x1, o_y1]) / self.image_size)  # scale to 0-1
                all_boxes.append(torch.tensor([min(s_x0, o_x0), min(s_y0, o_y0),
                                               max(s_x1, o_x1), max(s_y1, o_y1)]) / self.image_size)  # scale to 0-1
                all_masks.append(1)
                all_text_embeddings.append(anno["subject_" + text_embedding_name])
                all_text_embeddings.append(anno["object_" + text_embedding_name])
                all_text_embeddings.append(anno["action_" + text_embedding_name])
                # all_text_embeddings.append(anno["action_" + text_embedding_name])
                all_image_embeddings.append(self.mapping(anno["subject_" + image_embedding_name]))
                all_image_embeddings.append(self.mapping(anno["object_" + image_embedding_name]))
                all_image_embeddings.append(self.mapping(anno["action_" + image_embedding_name]))
                # all_action_image_embeddings.append(self.mapping(anno["action_" + image_embedding_name]))

        # Sort according to area and choose the largest N objects
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data*3, 4)
        masks = torch.zeros(self.max_boxes_per_data*1)
        text_embeddings = torch.zeros(self.max_boxes_per_data*3, self.embedding_len)
        image_embeddings = torch.zeros(self.max_boxes_per_data*3, self.embedding_len)

        # all_boxes = torch.stack(all_boxes)
        # all_text_embeddings = torch.stack(all_text_embeddings)
        # all_image_embeddings = torch.stack(all_image_embeddings)

        for i, idx in enumerate(wanted_idxs):
            boxes[i*3:i*3+3] = torch.stack(all_boxes[idx*3:idx*3+3])
            masks[i] = all_masks[idx]
            text_embeddings[i*3:i*3+3] = torch.stack(all_text_embeddings[idx*3:idx*3+3])
            image_embeddings[i*3:i*3+3] = torch.stack(all_image_embeddings[idx*3:idx*3+3])

        image_masks = masks
        text_masks = masks

        # if self.random_drop_embedding != 'none':
        #     image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        # else:
        #     image_masks = masks
        #     text_masks = masks

        out["boxes"] = boxes
        out["masks"] = masks  # indicating how many valid objects for this image-text data
        out["image_masks"] = image_masks  # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks  # indicating how many objects still there after random dropping applied
        out["text_embeddings"] = text_embeddings
        out["image_embeddings"] = image_embeddings

        # -------------------- caption ------------------- #
        if random.uniform(0, 1) < self.prob_use_caption or len(wanted_idxs) == 0:
            out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""

        return out

    def __len__(self):
        return len(self.files)

