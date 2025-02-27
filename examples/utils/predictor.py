# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torchvision.transforms import Resize
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, CfgNode


class ModPredictor:
    def __init__(self, cfg):
        if isinstance(cfg, CfgNode): 
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = Resize(cfg.INPUT.MIN_SIZE_TEST, max_size=cfg.INPUT.MAX_SIZE_TEST)
            self.input_format = cfg.INPUT.FORMAT
        else:
            self.cfg = cfg.copy()  # cfg can be modified by model
            self.model = instantiate(self.cfg.model).to(self.cfg.model.device) # type: ignore
            checkpointer = DetectionCheckpointer(self.model) # type: ignore
            checkpointer.load(self.cfg.weights)
            self.aug = Resize(self.cfg.dataloader.test.mapper.augmentation[0].short_edge_length, max_size=self.cfg.dataloader.test.mapper.augmentation[0].max_size)
            self.input_format = self.cfg.dataloader.test.mapper.img_format
            
        self.model.eval() # type: ignore
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, trand_y_tilde, original_height, original_width):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            image = trand_y_tilde[0]
            
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                image = image.flip(0)

            image = self.aug(image)

            inputs = {"image": image, "height": original_height, "width": original_width}
            predictions = self.model([inputs])[0]
            return predictions