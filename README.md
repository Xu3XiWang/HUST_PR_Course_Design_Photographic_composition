# HUST_PR_Course_Design_Photographic_composition
## DataSet
The dataset is KU_PCP_DATASETS, and it is a dataset include nine types of photographic composition.

http://mcl.korea.ac.kr/research/Submitted/jtlee_JVCIR2018/index.html

Jun-Tae Lee, Han-Ul Kim, Chul Lee, and Chang-Su Kim, "Photographic Composition Classification Photographs and Dominant Geometric Element Detection for Outdoor Scenes," Journal of Visual Communication and Image Representation, Aug. 2018.

## Training
```python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40 --optimizer "Adam"  --backbone "densenet"  --augmentation True```

you can use it to train the model. We can use alexnet, vgg16_bn, resnet(101), densenet(201) to work as backbone.
