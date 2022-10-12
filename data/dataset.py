import os
import csv
import torch
import xml.etree.ElementTree as et
from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, opt, phase, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.img_size = opt.input_size
        self.transform = transform
        self.S = opt.S
        self.B = opt.B
        self.C = opt.num_classes

        self.img_names = list()
        with open(os.path.join(self.data_dir, self.data_name, '{}.csv'.format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.img_names.append(line)

        self.label2num = {}
        with open(os.path.join(self.data_dir, self.data_name, 'label.txt'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.label2num[line.strip()] = i


    def __getitem__(self, index):
        # read image
        img = Image.open(os.path.join(self.data_dir, self.data_name, self.img_names[index][0]))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # read annotation (xml file)
        anno_path = os.path.join(self.data_dir, self.data_name, self.img_names[index][0].replace('jpg', 'xml'))
        anno_tree = et.parse(anno_path)
        anno_root = anno_tree.getroot()

        labels = []
        boxes = []

        for node in anno_root.findall("object"):
            obj_name = node.find("name").text
            obj_xmin = node.find("bndbox").find("xmin").text
            obj_ymin = node.find("bndbox").find("ymin").text
            obj_xmax = node.find("bndbox").find("xmax").text
            obj_ymax = node.find("bndbox").find("ymax").text   

            # labels
            labels.append(self.label2num[obj_name])

            # bound boxes
            bbox = [obj_xmin, obj_ymin, obj_xmax, obj_ymax]
            boxes.append(bbox)


        # resize bbox
        W = int(anno_root.find("size").find("width").text)
        H = int(anno_root.find("size").find("height").text)

        W_ratio = self.img_size/W
        H_ratio = self.img_size/H
        ratio_list = [W_ratio, H_ratio, W_ratio, H_ratio]
        resized_boxes = []

        for box in boxes:
            box = list(map(float, box))
            bbox = [a*b for a, b in zip(box, ratio_list)]

            # change from (x_min, y_min, x_max, y_max) to (x, y, w, h) format
            x = (bbox[0] + bbox[2])/2.0
            y = (bbox[1] + bbox[3])/2.0
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox = [x, y, w, h]
            resized_boxes.append(bbox)

        output_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        index = 0
        for box in resized_boxes:
            x, y, w, h = box
            cell_size = int(self.img_size / self.S)
            class_label = labels[index]

            # i, j represents the cell row and cell column
            i = y // cell_size
            j = x // cell_size
            x_cell = float(x/cell_size) - j
            y_cell = float(y/cell_size) - x
            w_cell = w / self.img_size
            h_cell = h / self.img_size

            if output_matrix[i, j, self.C] == 0:
                # Set that there exitsts an object
                output_matrix[i, j, self.C + 5 * index] = 1

                # Set box coordinates
                box_coord = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                box_position = self.C + 1 + 5 * min(index, 2)
                output_matrix[i, j, box_position : box_position + 4] = box_coord

                # Set one-hot encoding for class_label
                output_matrix[i, j, class_label] = 1

            index += 1
        
        return img, output_matrix


    def __len__(self):
        return len(self.img_names)