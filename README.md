# Audio Classification

## Dependency Setup

### Create new conda virtual environment

    conda create --name audio_classify python=3.7 -y
    conda activate audio_classify


### Installation

<pre>
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch -y
git clone 
pip install -r requirements.txt
<pre>

## Dataset Preparation
**Open source audio dataset**
Tomofun-AI 狗音辨識: https://github.com/lawrencechen0921/Tomofun-AI-

Kaggle Audio Cats and Dogs: https://www.kaggle.com/mmoreaux/audio-cats-and-dogs

Kaggle Freesound General-Purpose Audio Tagging Challenge: https://www.kaggle.com/c/freesound-audio-tagging/data

**Data Preprocessing**
If you want to try your dataset, please prepare the following items.
* The training/ validation dataset file
* The data label csv
* The dataset yaml

Take the Kaggle Audio Cats and Dogs dataset as an example, please place the dataset in different folders according to the category.

![image](https://github.com/chingi071/Audio_Classification/tree/main/README_pix/data_file.jpg)

Next, create the data label csv using the following ipynb file.

* create_data_csv.ipynb

Third, create the dataset yaml.

* cat_dog.yaml

![image](https://github.com/chingi071/Audio_Classification/tree/main/README_pix/cat_dog_yaml.jpg)

Take the Tomofun-AI dataset as an example, please do data preprocessing. You will get tomofun_train.csv.

* Tomofun_data_preprocessing.ipynb

And then create the dataset yaml.

![image](https://github.com/chingi071/Audio_Classification/tree/main/README_pix/tomofun_yaml.jpg)

The Tomofun-AI dataset structure is as follows:

<pre>
train
├── train_00001.wav
├── train_00002.wav
├── ...
└── train_01200.wav
tomofun_train.csv
<pre>

**Data Augmentation**
We use [Audiomentations](https://github.com/iver56/audiomentations) to add more data.
* data_augmentation.ipynb

The dataset structure is as follows:

<pre>
tomofun_aug_train
├── aug_0_train_00001.wav
├── aug_0_train_00002.wav
├── ...
├── train_00001.wav
├── train_00002.wav
├── ...
└── train_01200.wav
tomofun_aug_train.csv
<pre>

**Data Visualize**
* data_visualize.ipynb

## Training
The model you can choose: ResNet18、ResNet34、ResNet50、ResNet101、ResNet152、SENet、DenseNet、Convnext_tiny、Convnext_small、Convnext_base、Convnext_large

**Train on one GPU**

<pre>
python train.py --yaml_file=tomofun.yaml --model=ResNet18 --model_saved_path=workdirs
<pre>

**Train on multi-GPU**

<pre>
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --yaml_file=tomofun.yaml --model=Convnext_tiny --model_saved_path=workdirs
<pre>

To enable one more multi-GPU training, use the following command.

<pre>
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 9999 train.py --yaml_file=tomofun.yaml --model=Convnext_tiny --model_saved_path=workdirs
<pre>

**Start TensorBoard**

<pre>
tensorboard --logdir runs
<pre>

## Predict

<pre>
python predict.py --yaml_file=tomofun.yaml --model=Convnext_tiny --model_saved_path=workdirs --test_data=test_data
<pre>

## Convert to ONNX

<pre>
pip install onnx onnxruntime==1.6.0

python convert_to_onnx.py --yaml_file=tomofun.yaml --model=Convnext_tiny --model_saved_path=workdirs --model_weights=best.pth

python onnx_predict.py --test_data=test_data
<pre>

## Record audio

<pre>
pip install pyaudio
<pre>

Create the record file using the following ipynb file.

* record.ipynb

## Result

<pre>
device: cuda:1, rank: 1, world_size: 2
device: cuda:0, rank: 0, world_size: 2
Train_Epoch: 0/99, Training_Loss: 0.011717653522888819 Training_acc: 0.42
Train_Epoch: 0/99, Training_Loss: 0.012225324138998985 Training_acc: 0.40               
Valid_Epoch: 0/99, Valid_Loss: 0.010406222939491273 Valid_acc: 0.49
Valid_Epoch: 0/99, Valid_Loss: 0.01043313001592954 Valid_acc: 0.48
--------------------------------
Train_Epoch: 1/99, Training_Loss: 0.00876050346220533 Training_acc: 0.54               
Train_Epoch: 1/99, Training_Loss: 0.008517718284080426 Training_acc: 0.56               
Valid_Epoch: 1/99, Valid_Loss: 0.008887257364888986 Valid_acc: 0.57               
Valid_Epoch: 1/99, Valid_Loss: 0.008429310657083989 Valid_acc: 0.58               
--------------------------------                          

............

Train_Epoch: 99/99, Training_Loss: 4.295512663895462e-06 Training_acc: 1.00               
Valid_Epoch: 99/99, Valid_Loss: 0.0004894535513647663 Valid_acc: 0.99               
Train_Epoch: 99/99, Training_Loss: 2.0122603179591654e-06 Training_acc: 1.00               
Valid_Epoch: 99/99, Valid_Loss: 0.0006921298647505341 Valid_acc: 0.99             
--------------------------------
Finished Training.

<pre>

* Accuracy

![image](https://github.com/chingi071/Audio_Classification/tree/main/README_pix/Accuracy.jpg)

* Loss

![image](https://github.com/chingi071/Audio_Classification/tree/main/README_pix/Loss.jpg)

* Confusion Matrix

![image](https://github.com/chingi071/Audio_Classification/tree/main/README_pix/confusion_matrix.jpg)