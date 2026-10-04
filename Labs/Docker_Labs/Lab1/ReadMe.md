# Docker Lab 1: Breast Cancer Classifier

This lab trains a machine learning model inside a Docker container. The original lab trained one Random Forest on the Iris dataset and saved the model inside the container, so the model was lost once the container stopped. I changed it to use a different dataset, compare a few models, print proper evaluation metrics, and save the results back to my computer using a volume.

## What I changed

1. **Dataset:** I switched from Iris to the Breast Cancer Wisconsin dataset (built into sklearn). It has 569 samples and 30 features, and the goal is to predict if a tumor is malignant or benign.
2. **Comparing models:** Instead of training only one model, the script now trains three and compares them:
   - Logistic Regression (with StandardScaler in a pipeline)
   - Random Forest
   - Gradient Boosting
3. **Evaluation:** The original code did not test the model at all. Now it prints accuracy, malignant recall and ROC AUC for every model, and a full classification report for the best one.
4. **Choosing the best model:** I pick the best model by **malignant recall** first and use accuracy only as a tie breaker. For cancer detection, missing a malignant tumor (false negative) is much worse than a false alarm, so I think recall on the malignant class matters more than plain accuracy.
5. **Stratified split:** The train/test split keeps the same malignant/benign ratio in both sets.
6. **Saving output with a volume:** The best model (`cancer_model.pkl`) and a `metrics.json` file are saved to `/app/output`. I added `ENV OUTPUT_DIR` and `VOLUME` to the dockerfile so this folder can be mounted to my machine and the files don't get lost when the container exits.

## Files

```
Lab1/
├── dockerfile
├── ReadMe.md
├── output/              # created when you run with the volume
│   ├── cancer_model.pkl
│   └── metrics.json
└── src/
    ├── main.py
    └── requirements.txt
```

## How to run

Build the image:

```
docker build -t lab1:v2 .
```

Run it and mount the output folder (PowerShell on Windows):

```
docker run --rm -v "${PWD}/output:/app/output" lab1:v2
```

On Mac/Linux:

```
docker run --rm -v "$(pwd)/output:/app/output" lab1:v2
```

`--rm` removes the container after it finishes, and `-v` connects the `output` folder on my computer to `/app/output` inside the container. After it runs, `cancer_model.pkl` and `metrics.json` show up in the `output` folder.

You can also run it without the volume (`docker run lab1:v2`), it still trains and prints the results but the files stay inside the container.

To save the image as a tar file:

```
docker save lab1:v2 > my_image.tar
```

## Results

| Model | Accuracy | Malignant Recall | ROC AUC |
|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9762** | **0.9954** |
| Random Forest | 0.9561 | 0.9286 | 0.9937 |
| Gradient Boosting | 0.9561 | 0.9048 | 0.9907 |

Logistic Regression was the best model. It caught 41 out of 42 malignant tumors in the test set.

Classification report for the best model:

```
              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

## What I learned

- I expected Random Forest or Gradient Boosting to win since they are usually stronger models, but Logistic Regression did better on every metric. I think this is because after scaling, the data is close to linearly separable, so a simple model is enough. It showed me it's better to compare models than to assume the complex one is best.
- Gradient Boosting had the same accuracy as Random Forest but missed more cancers (4 vs 3). This is why looking only at accuracy can be misleading.
- Putting the scaler inside a pipeline means it gets saved together with the model, so the same scaling is applied when the model is used later.
- Anything a container writes is gone when the container is removed unless you use a volume. Code changes also need a rebuild because the code is copied into the image at build time.
