# EnsembleBench Notebooks

This directory contains notebooks utilizing the EnsembleBench library to perform ensembles of multiple pre-trained models. It also includes various diversity-based metrics for evaluation and supports different voting strategies for ensembling.

To evaluate individual or ensemble performance, modify the `model_names` variable in `inference.ipynb`:

```python
model_names = ['resnet18', 'resnet34']  # Modify this line to select desired models
```
Ensure that the `model_paths` variable is updated to match the selected models:
```python
model_paths = {
    'resnet18': r'./resnet18_best_model.pth',
    'resnet34': r'./resnet34_best_model.pth'
}
```
## Model Training Notebooks

The `modelTraining` directory contains notebooks that were used to train and validate MNIST and FashionMNIST datasets. 

## Example Comparison (FashionMNIST)

### Individual Model Accuracies:

| Index | Model         | Testing Accuracy (%) |
|-------|---------------|----------------------|
| 0     | ResNet34      | 94.18                |
| 1     | ResNet50      | 93.66                |
| 2     | ResNet101     | 93.65                |
| 3     | ResNet152     | 93.50                |
| 4     | AlexNet       | 92.22                |
| 5     | DenseNet121   | 94.51                |
| 6     | DenseNet161   | 94.60                |
| 7     | DenseNet169   | 94.73                |
| 8     | SqueezeNet1_1 | 93.01                |
| 9     | GoogleNet     | 94.52                |
| 10    | VGG11         | 92.14                |
| 11    | VGG13         | 92.27                |
| 12    | ConvNeXt Tiny | 94.05                |


### Ensemble Accuracies:

| Members (Indices) | Val Accuracy (%) | Testing Accuracy (%) |
|-------------------|------------------|----------------------|
| 4, 5, 6, 9        | 95.50            | 95.08                |
| 5, 6, 9, 10       | 95.44            | 95.17                |
| 2, 4, 7, 9, 10    | 95.42            | 95.11                |

<!-- ## Inference and Ensemble Selection -->

The inference notebook (`inference.ipynb`) loads weight files from models pretrained in the training notebook.

<!-- Ensemble selection logic is detailed in `FocalDiversityBasedEnsembleSelection.ipynb`. -->

<!-- ### Final Code Cell from `FocalDiversityBasedEnsembleSelection.ipynb`:

```python


print('EQ calculation')
EQ_members = teamSelectedFQAllDict[top_3_dm_acc[0][0]] & teamSelectedFQAllDict[top_3_dm_acc[1][0]] & teamSelectedFQAllDict[top_3_dm_acc[2][0]]
common_accuracies = [teamAccuracyDict[member] for member in EQ_members if member in teamAccuracyDict]

if common_accuracies:
    statistics = getNTeamStatistics(list(EQ_members), teamAccuracyDict, minAcc, avgAcc, maxAcc, tmpAccList)
    print("Statistics:", statistics)
else:
    print("No common members found or no accuracy data available.")
```
Output:

Baseline
(8178, 93.32500457763672, 95.5916748046875, 95.208412812257, 0.18567166295272966, 8157, 7651, 8171, 8178)
EQ calculation
Statistics: (25, 94.86666870117188, 95.50000762939453, 95.26600494384766, 0.1465713939803493, 24, 24, 25, 25)


Top EQ_members with their accuracies:
- ('4,5,6,9', 95.50000762939453)
- ('5,6,9,10', 95.4416732788086)
- ('2,4,7,9,10', 95.41667175292969)
 -->

Links to the weight files and predictions obtained after training are included inside the notebooks.