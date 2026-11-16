# Regression and Classification with PyTorch, deployed with ONNX

Homework by Ashish Julakanti. Two complete machine-learning pipelines in PyTorch, each comparing five models, with the best model of each exported to ONNX and served in the browser from GitHub Pages using ONNX Runtime Web.

**Live demo:** `https://<your-github-username>.github.io/<repo-name>/`

## Results (held-out test sets)

| Task | Dataset | Best model (chosen on validation) | Test result |
| --- | --- | --- | --- |
| Regression | California Housing (20,640 block groups) | ResidualNet | R² = 0.820, RMSE = $48,803, MAE = $33,446 |
| Classification | IBM Telco Customer Churn (7,043 customers) | ResidualNet, threshold 0.56 | F1 (churn) = 0.635, recall = 0.726, precision = 0.564, accuracy = 0.778, ROC-AUC = 0.843 |

Models compared in both parts: Linear / Logistic regression, MLP, DeepNN, F1plusF2 (linear + nonlinear) and ResidualNet.

## Repository layout

```
index.html                  web page (GitHub Pages entry point)
index.js                    encodes the form, runs the ONNX models with onnxruntime-web
housing_regressor.onnx      regression model   (input1: [batch, 13] raw features -> output1: [batch, 1] USD)
churn_classifier.onnx       classification model (input1: [batch, 26] -> output1: [batch, 2] = [P(stay), P(churn)])
notebooks/
  Part1_Regression_CaliforniaHousing.ipynb   full pipeline, executed with outputs
  Part2_Classification_TelcoChurn.ipynb      full pipeline, executed with outputs
  figures/                  every plot used in the report
  ONNXmodels/               ONNX exports + summary JSON of all metrics
data/                       copies of both CSVs (the notebooks download them from GitHub)
```

## Deploy to GitHub Pages

1. Create a new public repository on GitHub and upload everything in this folder (keep `index.html` at the top level).
2. In the repository go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to *Deploy from a branch*, choose `main` and `/ (root)`, then **Save**.
4. After a minute or two the page is live at `https://<your-github-username>.github.io/<repo-name>/`.

The page must be served over http(s); opening `index.html` directly from disk will not load the `.onnx` files. To test locally run `python -m http.server` in this folder and open `http://localhost:8000`.

## Re-running the notebooks

Open either notebook in Google Colab or Jupyter and run all cells (`pip install -r requirements.txt` if needed). Each notebook downloads its dataset, trains all five models (a few minutes on CPU), saves the figures and writes the ONNX file to `notebooks/ONNXmodels/`. Copy the new `.onnx` file next to `index.html` to update the web page.

## Data sources

- California Housing: https://github.com/ageron/handson-ml2/tree/master/datasets/housing
- Telco Customer Churn: https://github.com/IBM/telco-customer-churn-on-icp4d
- ONNX deployment pattern: https://github.com/rcalix1/ConstraintsMLprediction/tree/main/Deployment/RealData/Run10
