# Deep Learning Mini Project1

## Hyperparameter Tuning on CIFAR10 Dataset using ResNet18

We sought to find the best performing model using less than 5 million parameters on CIFAR-10 test set data with ResNet architecture while modifying 6 architecture specific parameters, doing general hyperparameter tuning, and performing transformations. We first varied the 6 architecture specific parameters one a time keeping other parameters fixed at ResNet18 values. Then based on our understanding of how each parameter affects the final model, we chose a subset of values for each parameter to test in a joint optimization. We ran the best performing parameter assignment from the joint optimization for 500 epochs, achieving a final test set accuracy of 92.5\%. For general hyperparameter tuning and transformations, we selected values and transformations successful in the literature. The ResNet architecture parameter assignment was N = 3, $B_i = 2$, $C_i = 85$, $F_i = 3$, P = 8, $K_i = 1$. We used ADAM optimizer with $\beta_1 = 0.9$ and $\beta_2 = 0.999$, a learning rate scheduler where we start with lr=0.001 and decay it 10\% every 10 epochs, batch size 64, and various transformations to train our network


## Final Results
Below you can see out final architecture and results. 

![network drawio](https://user-images.githubusercontent.com/53308177/160037370-96eced68-68cd-412e-bc8b-6a8a95a8a5a4.png)

![fin](https://user-images.githubusercontent.com/53308177/160037640-247ced74-e6b6-472d-b300-df81af5f4c3c.png)

## How to Run the Code

Parametric_Base_Code.ipynb: Our parametric code in which we can change the tuning parameters without making any other changes in the model. All the adjustments (stride, dimensions, etc) are pre-handled.

Long_Run_9_Batch_64_Fast.ipynb: Our final architecture, joint optimization code. 






