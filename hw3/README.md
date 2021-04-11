# Skeleton Code for Deep Learning Homework 3

### Get started
```
git clone https://github.com/IrisLi17/dl_course_hw3.git
cd dl_course_hw3
# If you are using conda
conda env create -f environment.yml
```
Download the dataset here https://cloud.tsinghua.edu.cn/d/c7572d5c3c7b4c0cb8bd/ and organize your folder as follows:
```
dl_course_hw3
├── data
    └── MNIST
        ├── processed
        └── raw
├── ebm
├── flow
├── gan
├── vae
└── README.md
```
Energy-based model:
```
cd ebm
python main.py [optional arguments]
# Play around after you have trained a model.
python main.py --play --load_dir /path/to/your/saved/model [optional arguments]
```
Flow:
```
cd flow
python train.py
# Try image inpainting.
python inpainting.py
```
VAE:
```
cd vae
python vae.py [optional arguments]
# Generate samples for further evaluation.
python vae.py --eval --load_path /path/to/your/saved/model [optional arguments]
```
GAN:
```
cd gan
python gan.py [optional arguments]
# Generate samples for further evaluation.
python gan.py --eval --load_path /path/to/your/saved/model [optional arguments]
```