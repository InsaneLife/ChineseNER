# NER
中文命名实体识别


## 底层编码器
- IDCNN
- BiRNN
- Transformer
- 

TODO: 
1. BiRNN + CRF
2. Linguistic Features
   1. POS Tags
   2. Dependency Labels
   3. NER Labels
   4. Chunking Labels
   5. Brown Clustering Labels
3. Attention
4. Bert

# data
## MSRA NER微软亚洲研究院数据集。
  - 5 万多条中文命名实体识别标注数据（包括地点、机构、人物） 
  - https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA


# environment 
tensorflow = 1.8
python = 3.5
cuda = 9.0

# reference
1. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) 
2. [Neural Sequence Labeling with Linguistic Features](https://www.aclweb.org/anthology/S18-1114) 
3. 