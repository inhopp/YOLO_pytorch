import os
import torch
import torch.optim
import torch.nn as nn

from tqdm import tqdm
from data import generate_loader
from option import get_option
from model import Yolov1
from loss import YoloLoss
from utils.bbox_tools import get_bboxes
from utils.mAP import mAP

class Solver():
    def __init__(self, opt):
        self.dev = torch.device("cuda: {}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.model = Yolov1(opt).to(self.dev)
        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, opt.data_name, "best_epoch.pt")
            self.model.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.device_ids).to(self.dev)

        print("# params:", sum(map(lambda x: x.numel(), self.model.parameters())))
        self.loss_fn = YoloLoss(opt)
        self.optim = torch.optim.Adam(self.model.parameters(), opt.lr, weight_decay=opt.weight_decay)

        self.train_loader = generate_loader('train', opt)
        print("train set ready")
        self.val_loader = generate_loader('test', opt)
        print("validation set ready")
        self.best_mAP, self.best_epoch = 0, 0

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            self.model.train()
            loop = tqdm(self.train_loader, leave=True)
            mean_loss = []

            for _, (img, output_label) in enumerate(loop):
                img = img.to(self.dev)
                output_label = output_label.to(self.dev)
                preds = self.model(img)
                loss = self.loss_fn(preds, output_label)
                mean_loss.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loop.set_postfix(loss=loss.item())
            
            # eval_epoch = 1
            pred_boxes, target_boxes = get_bboxes(
                opt, self.val_loader, self.model, iou_threshold=0.5, threshold=0.4)

            mean_avg_prec = mAP(pred_boxes, target_boxes, iou_threshold=0.5, num_classes=opt.num_classes)

            if mean_avg_prec >= self.best_mAP:
                self.best_mAP, self.best_epoch = mean_avg_prec, epoch
                self.save()

            print("Epoch [{}/{}] Loss: {:.3f}, Val mAP: {:.3f}".format(epoch+1, opt.n_epoch, sum(mean_loss)/len(mean_loss), mean_avg_prec))
            print("Best : {:.2f} @ {}".format(self.best_mAP, self.best_epoch+1))



    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, "best_epoch.pt")
        torch.save(self.model.state_dict(), save_path)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()


if __name__ == "__main__":
    main()