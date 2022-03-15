# [WeChat Big Data Challenge](https://algo.weixin.qq.com/) Program Share (Ranking: 178/1614)

Team: 田老师我不会

## Idea

1. Base Model--DeepFM
2. The number of negative samples is far more than positive samples: like(4%), click_avatar(0.7%), forward(0.3%), comment(0.04%)
   - Sample negative according to ratio--only for single model
   - Sample negative according to play time
3. Four targets has a certain extent correlations--MMOE
4. Whether complete a video reflect users’ preference--Add a task to indicate whether a user complete a video.

The idea is to build a multi-task model based on xDeepFM. Click Duration (CD) produces after click, whose relation is similar to CTR and CVR. Therefore, we use ESMM model CTR and CTCD respectively, in which CTCD modeling is an auxiliary task to help promote CTR prediction. At last, output only contains prediction of CTR.

## Code Structure

```bash
├── data # Data Preprocessing Module
│   ├── DA.ipynb 
│   ├── preprocess.ipynb # Data preprocessing code
├── submit	# save results
├── generator.py # DataSet class for single task
├── Grid_Search.py  # grid search hyper-parameters
├── log # Log Module
│   ├── tensorboard
│   └── text
├── mainMT.py  # Main function of ESMM
├── main.py # Main Function of single task
├── submitMT.py  # predict and generate submission for multi-task
├── submit.py # predict and generate submission
├── README.md
├── config.ini	# arguments for training
├── run.bash # run script
├── submission
│   ├── average.ipynb # average n runs 
└──  src
	├── models # Model Module
	│   ├── basemodel.py # Base model
	│   ├── baseMT.py	# Multi-task Base model
	│   ├── deepfm.py	# DeepFM Model
	│   └── mmoe.py  # MMOE model
	└── utils # Tools
    	├── evaluation.py
    	├── selection.py
    	└── utils.py
```

## Run

1. Unzip all dataset and move to `\data\`. Then, run preprocess.ipynb.
2. Train model

```
bash run.bash
```

All arguments are defined in mainMT.py。
The optimal group of hyper-parameters is:

```bash
batch_size=2048
learning_rate=0.001
epoch=1
embedding_size=32
```

## Environments

```bash
pip install -r requirements.txt
```
