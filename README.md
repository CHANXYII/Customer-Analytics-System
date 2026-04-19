# Customer Analytics System

A comprehensive Customer Analytics System that integrates Autoencoder, Semi-Supervised Learning, KMeans Segmentation, and a Recommendation System. This project transforms raw customer survey data into actionable marketing insights, presented through a responsive Fast-API web application.

## 🚀 Features

- **End-to-End ML Pipeline**: Seamless integration of Data Preprocessing, Dimensionality Reduction, Clustering, and Recommendations.
- **Autoencoder**: Compresses high-dimensional Likert scale survey responses into core latent factors to filter out noise.
- **Semi-Supervised Learning (SSL)**: Uses `LabelSpreading` to infer unknown marketing intents from a sparse subset of known customer intents.
- **KMeans Segmentation**: Groups customers into distinctive marketing segments based on their latent behavioral traits.
- **Real-Time Web Interface**: A modern UI with interactive sliders allowing users to emulate survey answers and get instantaneous marketing campaigns (Persona Group & Suggested Insights).

## 🛠 Tech Stack

- **Backend / API**: Python, FastAPI, Uvicorn
- **Machine Learning**: Scikit-Learn, TensorFlow/Keras, Pandas, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## 📂 Project Structure

```text
📦 Customer-Analytics-System
 ┣ 📂 static
 ┃ ┗ 📜 index.html              # Modern, glowing UI for user input and insights
 ┣ 📜 main.py                   # FastAPI application and inference entry point
 ┣ 📜 ml_pipeline.py            # Complete ML training & data processing pipeline
 ┣ 📜 BU Data from Survey Cases_final.xlsx  # Raw survey dataset
 ┣ 📜 requirements.txt          # Python dependencies
 ┗ 📜 README.md                 # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CHANXYII/Customer-Analytics-System.git
   cd Customer-Analytics-System
   ```

2. **Activate the Virtual Environment:**
   If using `venv`:
   ```bash
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   Starts the FastAPI development server. The ML models are initialized and trained in-memory via `ml_pipeline.py` during startup.
   ```bash
   uvicorn main:app --port 8000 --reload
   ```

5. **Access the Web UI:**
   Open your browser and navigate to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 📊 How It Works (Pipeline)

1. **Data Load**: Ingestion of real-world survey responses (`.xlsx`).
2. **Dimension Reduction**: `train_autoencoder` extracts the most relevant behavioral traits using a neural network bottleneck and suppresses irrelevant noise.
3. **Intent Extrapolation**: `predict_interests_ssl` leverages a KNN kernel to predict missing intent labels.
4. **Customer Grouping**: `segment_customers` relies on KMeans algorithm to divide respondents into distinct buyer modes.
5. **Marketing Mapping**: Maps the final groups directly into actionable business strategies (e.g., *Cool Down Campaign*, *Stay Ready Campaign*).
