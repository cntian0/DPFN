# Dual-Perspective Fusion Network for Aspect-based Multimodal Sentiment Analysis

### Processing data

1、The twitter dataset is placed in the directory `data`, and the contents of each directory are as follows:：

```
data/twitter2015: Text data from the twitter2015 dataset
data/twitter2017: Text data from the twitter2017 dataset
data/twitter2015_images: Image data from the twitter2015 dataset
data/twitter2017_images: Image data from the twitter2017 dataset
```

2、The image feature data extracted using ViT is stored in the directory `data/imgDealFile`, and the command `python img_deal_by_vit.py` is executed to obtain this data.

3、The image caption needs to be obtained before model training, which is stored in the directory `data/caption`, and the extraction method taken in this paper is the same as [CapTrBERT](https://dl.acm.org/doi/10.1145/3474085.3475692).

4、Syntactic information is extracted using [LAL-Parser](https://github.com/KhalilMrini/LAL-Parser), and the extracted syntactic information is stored in the directory `data/oriAdj`, which again needs to be obtained before model training.

>We provide extracted image captions and syntactic information, which can be accessed at the following link:
>
>link: https://pan.baidu.com/s/1WtmKmuJyI-MGgcaUQ9UYIQ 
>extraction code: xbpg 

### Training Models

Run the following command to start model training:：

```
python train.py
```

