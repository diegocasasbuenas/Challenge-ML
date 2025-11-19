# ML Challenge - Documentation

## Model Selection

**Chosen Model:** Logistic Regression with class balancing

**Reasons:**
Since the logistic regression model showed results very similar to XGBoost, we selected logistic regression because it has a lower computational cost, which allows for better inference latency.

Additionally, I chose the version of the model trained with class balancing because it performs better when identifying class 1 (the positive class). This means it is more effective at detecting when a flight is likely to be delayed. Although this may introduce some false positives, predicting a delay is generally less costly than failing to detect one. I considered that misclassifying a delayed flight as “on time” can lead to higher operational costs and negatively impact customer experience, whereas predicting a delay when there isn’t one has a much lower impact.

**Class Balancing:**
```python
class_weight = {1: n_y0/len(target), 0: n_y1/len(target)}
```

## Features

Top 10 features (one-hot encoded):
- Airlines: Latin American Wings, Grupo LATAM, Sky Airline, Copa Air
- Months: 4, 7, 10, 11, 12
- Flight type: International (TIPOVUELO_I)

## Model Implementation (model.py)

The `DelayModel` class handles the complete ML pipeline. It loads a pre-trained model from disk on initialization to avoid training on every request. The `preprocess` method takes raw flight data and generates one-hot encoded features for airline, flight type, and month, then filters to keep only the top 10 most important features. Missing columns are filled with zeros to handle unseen categories. The `fit` method trains a LogisticRegression with custom class weights to handle the imbalanced dataset. The `predict` method returns a list of binary predictions (0 = on time, 1 = delayed).

## API Implementation (api.py)

The FastAPI application exposes two endpoints. The `/health` endpoint provides a simple status check. The `/predict` endpoint accepts a list of flights with OPERA, TIPOVUELO, and MES fields. Pydantic validators ensure MES is between 1-12 and TIPOVUELO is either "N" or "I", returning HTTP 400 for invalid inputs. The model is instantiated once at startup for efficiency, and each request preprocesses the input data and returns predictions as a JSON response.

## CI/CD Implementation

The CI pipeline (ci.yml) triggers on pushes to develop and feature branches, and on pull requests. It sets up Python 3.11, installs all dependencies, runs model tests and API tests, and uploads coverage reports as artifacts.

The CD pipeline (cd.yml) triggers on pushes to main. It authenticates to Google Cloud using a service account, builds the Docker image with the commit SHA as tag, pushes it to Artifact Registry, and deploys to Cloud Run with public access enabled.

## Deployment

- **Platform:** Google Cloud Run
- **URL:** https://challenge-api-595836143326.us-central1.run.app
- **Autoscaling:** 0-5 instances

## Stress Test Results

- 9,271 requests in 60s
- 0 failures
- 160 req/s
- Avg response: 184ms

**It could variates at every test**