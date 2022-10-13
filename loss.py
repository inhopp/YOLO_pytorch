import torch
import torch.nn as nn

from utils.IoU import IoU
from utils.nms import nms

class YoloLoss(nn.Module):

    def __init__(self, opt):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = opt.S
        self.B = opt.B
        self.C = opt.num_classes

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        # make [batch_size, ...]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Assuming B = 2 (in paper)
        iou_b1 = IoU(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        iou_b2 = IoU(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        max_iou, best_box = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3) #  I_obj_i (1 or 0)

        # bbox coordinates  #
        # if best_box = 1 : select second box
        # if best_box = 0 : select first box
        box_predictions = exists_box * (
            best_box * predictions[..., self.C+6:self.C+10] + (1 - best_box) * predictions[..., self.C+1:self.C+5]
            )
        box_targets = exists_box * target[..., self.C+1:self.C+5]

        # box_predictions could be negative because initialized weights
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=2), 
            torch.flatten(box_targets, end_dim=2)
        )

        #    object loss    #
        pred_box = (best_box * predictions[..., self.C+5:self.C+6] + (1 - best_box) * predictions[..., self.C:self.C+1])
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box), 
            torch.flatten(exists_box * target[..., self.C:self.C+1])
            )

        #  non-object loss  #
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        #    class loss     #
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        # Total loss
        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )

        return loss