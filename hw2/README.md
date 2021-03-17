##### Requirements:

- Python version == 3.6
- numpy == 1.19.2
- torch == 1.7.1
- torchvision == 0.8.0

##### Run the examples:

1. Download the dataset from https://cloud.tsinghua.edu.cn/d/00e0704738e04d32978b/ and organized the data as follows:

   ```css
   dl_course_hw2
   ├── README.md
   ├── data
       └── cifar_10_4x
           ├── train
           └── valid
   ├── cifar10_4x.py
   ├── evaluation.py
   ├── model.py
   └── train.py
   ```

2. Run the example:

   ``` bash
   python train.py
   ```

3. Evaluate your model:

   ```bash
   python evaluation.py
   ```

##### Instruction

1. Run the example.
2. Train your own model. 
3. Your final model should not be larger than 200M and using any pre-trained model is **NOT** permitted.
4. **DO NOT** change the file *evaluation.py* or *cifar10\_4x.py*, and make sure you can test your model using *evaluation.py*. 
5. Name your best model *cifar10\_4x\_best.pth*. Submit this single model file and all your .py files to weblearning. We will use *evaluation.py* to evaluate your model on the test set.
6. Submit your report. 

##### Grading

1. Regarding the evaluation criteria of your model, assume your  test accuracy is X then your score is

<div align=center><img src="https://latex.codecogs.com/svg.latex?\frac{\min(X,0.9)-0.6}{0.9-0.6}\times7"></div>

2. **Bonus**: The best submission with the highest testing accuracy will get 1 point for the final grade.

