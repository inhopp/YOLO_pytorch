import os
import torch
from model import Yolov1
from data import generate_loader
from option import get_option
from utils.nms import nms
from utils.bbox_tools import plot_image, cellboxes_to_boxes

def inference(opt):
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")

    data_loader = generate_loader('temp', opt)
    print("data load complete")

    model = Yolov1(opt).to(dev)
    load_path = os.path.join(opt.chpt_root, opt.data_name, "best_epoch.pt")
    model.load_state_dict(torch.load(load_path))
    print("model construct complete")

    for img, _ in data_loader:
        img = img.to(dev)
        
        for idx in range(opt.eval_batch_size):
            bboxes = cellboxes_to_boxes(model(img))
            nms_bboxes = nms(bboxes[idx], iou_threshold=0.5, class_threshold=0.4)
            plot_image(img[idx].to("cpu"), nms_bboxes)


if __name__ =='__main__':
    opt = get_option()
    torch.manual_seed(opt.seed)
    inference(opt)