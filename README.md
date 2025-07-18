# MindAI Sentiment Analysis API

This project is a sentiment analysis API built with FastAPI that provides sentiment predictions for text inputs. It includes a machine learning pipeline for training a sentiment analysis model, experiment tracking with MLflow, and a simple dashboard to visualize sentiment data.

## Features

- **Sentiment Analysis API**: A robust API to get sentiment predictions (positive/negative) from text.
- **Machine Learning Pipeline**: A complete pipeline to train a sentiment analysis model using Scikit-learn.
- **Experiment Tracking**: Integrated with MLflow to log and track model training experiments.
- **Dashboard**: A simple dashboard to visualize sentiment distribution.
- **Health Check**: An endpoint to monitor the status of the API and the model.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/mindai-sentiment-api.git
    cd mindai-sentiment-api
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the API

To run the FastAPI application, use the following command:

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Training the Model

To train the sentiment analysis model, run the following script:

```bash
python scripts/train_model.py
```

This script will load the data, train the model, save it to the `models/` directory, and log the experiment in MLflow.

## API Endpoints

-   **`POST /predict`**: Get sentiment prediction for a given text.
    -   **Request Body**:
        ```json
        {
          "text": "I love this product!",
          "user_id": "user123"
        }
        ```
    -   **Response**:
        ```json
        {
          "sentiment": "positive",
          "confidence": 0.95,
          "interventions": [
            "Keep up the good mood!",
            "Share your positivity with others",
            "Stay consistent with your routine"
          ],
          "timestamp": "2023-10-27T12:00:00.000Z"
        }
        ```
-   **`GET /health`**: Check the health of the API.
-   **`GET /`**: View the sentiment analysis dashboard.

## Project Structure

```
.
├── .devcontainer/
├── data/
│   └── sentiment140.csv
├── models/
│   └── sentiment_model.pkl
├── scripts/
│   └── train_model.py
├── src/
│   ├── api.py
│   ├── data_processor.py
│   └── model_trainer.py
├── templates/
│   └── dashboard.html
├── .gitignore
├── README.md
└── requirements.txt
```
