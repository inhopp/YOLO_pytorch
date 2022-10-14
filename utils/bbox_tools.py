import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.nms import nms


def plot_image(image, boxes):
    # Plots predicted bboxes on the image
    im = np.array(image)
    h, w, _ = im.shape

    fig, ax = plt.subplot(1)
    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        x_min = box[0] - box[2] / 2
        y_max = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (x_min * w, y_max * h),
            box[2] * w,
            box[3] * h,
            linewidth=1,
            edgecolor="r",
            facecolor="none"
        )

        ax.add_patch(rect)
    plt.show()


def get_bboxes(opt, loader, model, iou_threshold, threshold):
    device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels, opt)
        bboxes = cellboxes_to_boxes(predictions, opt)

        for idx in range(batch_size):
            nms_boxes = nms(bboxes[idx], iou_threshold=iou_threshold, class_threshold=threshold)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, opt):
    S = opt.S
    B = opt.B
    C = opt.num_classes

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)
    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6:C+10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, opt):
    S = opt.S

    converted_pred = convert_cellboxes(out, opt).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes