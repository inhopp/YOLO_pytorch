import torch
from IoU import IoU

def nms(bboxes, iou_threshold, class_threshold):
    # bboxes: list of lists containing all bboxes with each bboxes 
    # specified as [classes_pred, prob_score, x, y, w, h] format
    bboxes = [box for box in bboxes if box[1] > class_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes 
            if box[0] != chosen_box[0] 
            or IoU(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms