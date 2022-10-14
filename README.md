# YOLO_pytorch

> [Paper_Review](https://inhopp.github.io/paper/Paper11/)

> cell index가 19~20으로 튀는 데이터가 존재하는데 아직 해결x <br> something wrong (train/apple_61.jpg : i=19, j=20 ) 


<br>

## Repository Directory 

``` python 
├── YOLO_pytorch
        ├── datasets
        │     └── fruits
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── utils
        │      ├── bbox_tools.py
        │      ├── IoU.py
        │      ├── mAP.py
        │      └── nms.py      
        ├── option.py
        ├── model.py
        ├── loss.py
        ├── train.py
        └── inference.py
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get annotations
- `utils` : utils for models and pre/post processing
- `option.py` : Environment setting


<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
!git clone https://github.com/inhopp/YOLO_pytorch.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --data_name {}(default: fruits) \
    --lr {}(default: 2e-5) \
    --n_epoch {}(default: 10) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 16) \
```

### inference
```python
python3 inference.py
    --device {}(defautl: cpu) \
    --data_name {}(default: fruits) \
    --num_workers {}(default: 4) \
```

<br>

#### Main Reference
https://github.com/aladdinpersson/Machine-Learning-Collection