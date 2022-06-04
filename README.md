# Machine-learning-project-of-heart-with-MLops
---

![heart_disease](https://user-images.githubusercontent.com/30417399/172022607-21d36c61-786e-441d-a4e0-87512f032800.png)

available in: [link](https://www.alibabacloud.com/blog/predicting-heart-diseases-with-machine-learning_218458)

This project is a mlops aplication to produce predict the heart disease using one of the kaggle's [dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease?select=heart_2020_cleaned.csv). We gonna focus on the pipeline construction since the download till the train. But one thing we used to facility the reproductibilitie of this project is the poetry. so we gonna do just two topics in this readme: 
1. How to install and use it?
2. How is the mlops pipeline

## 1. How to install the project
The only thing you need to do is: install [python-poetry](https://python-poetry.org/) and pyenv., clone this repository where the poetry was installed, 
tip the command 
```!poetry install```
.The poetry.toml file present in this repository has all the packages this project uses to run, when you use poetry install all this packages are installed in a enviroment created by the poetry. If the install command don't work, tip ```poetry lock``` and create the dependencies file to resolve it.

## 2. the full pipeline 
the full pipeline was made using sklearn and weight and biases mlops tool. to know more about the pipeline construction you can read the [medium article](https://medium.com/@diego25rn/reproductible-mlops-pipeline-just-like-a-poetry-72223ea2954b) about it. there i explain mlops concepts and project choices i have made. But to see how the pipeline works i sugest to acess the folder source/mlop/modules, there you gonna find all the functions and modules constrcted to build this project. To ilustrate how the pipeline looks like i had made the next figure to show how does the pipeline works.

![Untitled Diagram drawio(3) drawio](https://user-images.githubusercontent.com/30417399/172023364-540c4ab8-cb64-4d9e-9cd9-4d8a75ab1eb6.png)


([out of topics](https://github.com/diego2017003/machine-learning-project-of-heart-api))to conitnue this project i started to build the api but doesn't work yet
