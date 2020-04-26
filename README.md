# Pedestrian-Attribute-ReID

The PyTorch implementation for jointly training of pedestrian attribute recognition and Re-ID. The model is evaluated on Market-1501(with attribute), DukeMTMC-reID(with attribute) datasets and achieve the state-of-the-art performance on the two tasks.

## Usage
Jointly training with some hyper-parameters.
```
python3 train_joint.py [--warm_epoch 10] [--stride 1] [--erasing_p 0.5] [--batch-size 8] [--lr 0.02] [--num-epoch 80]
```
Evaluation for the pedestrian attribute recognition task.
```
python3 test_joint.py [--stride 1] [--fusion] [--alpha 0.5]
```
Evaluation for the Re-ID task.
```
python3 test_reid.py --stride 1 --frame_junk
```
Visualize the similarity of positive and negative samples.
```
python3 prepare_threshold.py 
python3 threshold.py [--stride 1]
```
Recognize the pedestrian attributes in the given images.
```
python3  inference.py   path_to_image  [--stride 1]
```
## Performance

Evaluation for reID on Market-1501
```
(Camera junk) Rank@1:0.930523 Rank@5:0.975653 Rank@10:0.983373 mAP:0.816014
(Frame junk) Rank@1:0.964371 Rank@5:0.987233 Rank@10:0.992280 mAP:0.843044
```
Evaluation for reID on DukeMTMC-reID
```
(Camera junk) Rank@1:0.853680 Rank@5:0.925494 Rank@10:0.948384 mAP:0.707455
(Frame junk) Rank@1:0.965889 Rank@5:0.987881 Rank@10:0.991023 mAP:0.798274
```
Evaluation for pedestrian attribute recognition on Market-1501-attribute
```
+------------+----------+-----------+--------+----------+
| attribute  | accuracy | precision | recall | f1 score |
+------------+----------+-----------+--------+----------+
|   young    |  0.998   |   0.000   | 0.000  |  0.000   |
|  teenager  |  0.873   |   0.938   | 0.915  |  0.926   |
|   adult    |  0.886   |   0.526   | 0.553  |  0.539   |
|    old     |  0.994   |   0.000   | 0.000  |  0.000   |
|  backpack  |  0.877   |   0.818   | 0.656  |  0.728   |
|    bag     |  0.785   |   0.568   | 0.478  |  0.519   |
|  handbag   |  0.898   |   0.320   | 0.072  |  0.118   |
|  clothes   |  0.926   |   0.936   | 0.983  |  0.959   |
|    down    |  0.941   |   0.976   | 0.934  |  0.955   |
|     up     |  0.935   |   0.935   | 1.000  |  0.967   |
|    hair    |  0.893   |   0.867   | 0.832  |  0.849   |
|    hat     |  0.977   |   0.783   | 0.287  |  0.420   |
|   gender   |  0.931   |   0.945   | 0.895  |  0.919   |
|  upblack   |  0.953   |   0.900   | 0.734  |  0.809   |
|  upwhite   |  0.920   |   0.861   | 0.834  |  0.847   |
|   upred    |  0.972   |   0.889   | 0.832  |  0.859   |
|  uppurple  |  0.986   |   0.716   | 0.807  |  0.759   |
|  upyellow  |  0.976   |   0.921   | 0.801  |  0.857   |
|   upgray   |  0.911   |   0.835   | 0.419  |  0.558   |
|   upblue   |  0.948   |   0.882   | 0.443  |  0.590   |
|  upgreen   |  0.969   |   0.810   | 0.739  |  0.773   |
| downblack  |  0.889   |   0.854   | 0.859  |  0.857   |
| downwhite  |  0.960   |   0.677   | 0.508  |  0.580   |
|  downpink  |  0.988   |   0.846   | 0.664  |  0.744   |
| downpurple |  1.000   |   0.000   | 0.000  |  0.000   |
| downyellow |  0.999   |   0.000   | 0.000  |  0.000   |
|  downgray  |  0.873   |   0.765   | 0.393  |  0.519   |
|  downblue  |  0.862   |   0.782   | 0.430  |  0.555   |
| downgreen  |  0.973   |   1.000   | 0.022  |  0.044   |
| downbrown  |  0.962   |   0.779   | 0.625  |  0.694   |
+------------+----------+-----------+--------+----------+
Average accuracy: 0.9351
Average f1 score: 0.5982
```
Evaluation for pedestrian attribute recognition on DukeMTMC-reID-attribute
```
+-----------+----------+-----------+--------+----------+
| attribute | accuracy | precision | recall | f1 score |
+-----------+----------+-----------+--------+----------+
|  backpack |  0.813   |   0.808   | 0.862  |  0.835   |
|    bag    |  0.807   |   0.402   | 0.366  |  0.383   |
|  handbag  |  0.934   |   0.170   | 0.008  |  0.015   |
|   boots   |  0.908   |   0.808   | 0.771  |  0.789   |
|   gender  |  0.873   |   0.842   | 0.823  |  0.832   |
|    hat    |  0.910   |   0.862   | 0.756  |  0.806   |
|   shoes   |  0.922   |   0.760   | 0.480  |  0.588   |
|    top    |  0.866   |   0.447   | 0.412  |  0.429   |
|  upblack  |  0.825   |   0.853   | 0.870  |  0.862   |
|  upwhite  |  0.957   |   0.775   | 0.428  |  0.552   |
|   upred   |  0.974   |   0.784   | 0.602  |  0.681   |
|  uppurple |  0.996   |   0.077   | 0.015  |  0.026   |
|   upgray  |  0.903   |   0.643   | 0.332  |  0.437   |
|   upblue  |  0.943   |   0.779   | 0.502  |  0.611   |
|  upgreen  |  0.975   |   0.462   | 0.350  |  0.398   |
|  upbrown  |  0.980   |   0.459   | 0.212  |  0.290   |
| downblack |  0.772   |   0.741   | 0.753  |  0.747   |
| downwhite |  0.942   |   0.790   | 0.317  |  0.453   |
|  downred  |  0.988   |   0.684   | 0.436  |  0.532   |
|  downgray |  0.917   |   0.355   | 0.212  |  0.265   |
|  downblue |  0.778   |   0.709   | 0.598  |  0.649   |
| downgreen |  0.997   |   0.000   | 0.000  |  0.000   |
| downbrown |  0.976   |   0.892   | 0.499  |  0.640   |
+-----------+----------+-----------+--------+----------+
Average accuracy: 0.9111
Average f1 score: 0.5139
```

## Update
- 2019-12-4: upload the project  
- 2019-12-9: add new transforms and fea_cat  
- 2019-12-14: 1. add tricks: stride, erasing_p, warm_epoch, lr 2. add training option: continue or restart 3. fix the bug of image size among training and testing files 4. transform the dimension of features from 2048 to 512: remove classifier_re
id.classifier but not classifier_reid  
- 2019-12-16: add --frame_junk  
- 2020-1-12: add --fusion --alpha  
- 2020-1-13: add threshold evaluation  

## Related Repos
[Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)

[Person-Attribute-Recognition-MarketDuke](https://github.com/hyk1996/Person-Attribute-Recognition-MarketDuke) ![GitHub stars](https://img.shields.io/github/stars/hyk1996/Person-Attribute-Recognition-MarketDuke.svg?style=flat&label=Star)
