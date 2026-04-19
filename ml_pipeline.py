import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

# Setting seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(filepath="BU Data from Survey Cases_final.xlsx"):
    """
    Load real survey data and preprocess numeric features (Likert scales 1-5).
    """
    df = pd.read_excel(filepath, header=1)
    
    numeric_cols = []
    for col in df.columns:
        if 'Timestamp' in col:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > (len(df) * 0.5): # Keep if mostly numeric
            df[col] = converted
            numeric_cols.append(col)
            
    # Extract only valid numeric Likert responses and fill any missing with a neutral 3.0
    numeric_df = df[numeric_cols].fillna(3.0)
    features = numeric_df.values

    # Simulate missing target labels for Semi-Supervised Learning
    n_samples = features.shape[0]
    labels = np.full(n_samples, -1)
    
    # We only 'know' 10% of our customers' true marketing intent
    labeled_subset = np.random.choice(n_samples, size=max(5, int(0.1 * n_samples)), replace=False)
    labels[labeled_subset] = np.random.randint(0, 3, size=len(labeled_subset))
    
    return features, labels

def train_autoencoder(X):
    """
    Step 1: Autoencoder.
    Compresses high-dimensional Likert scale survey responses down to core latent factors.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(5, activation='relu')(input_layer) # Bottleneck
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=16, verbose=0)
    
    latent_features = encoder.predict(X_scaled, verbose=0)
    return latent_features, encoder, scaler

def predict_interests_ssl(latent_features, labels):
    """
    Step 2: Semi-Supervised Learning.
    Expands known intents to the rest of the survey respondents using the latent map.
    """
    # knn algorithm ensures robust proximity mapping even without full convergence
    label_spread = LabelSpreading(kernel='knn', alpha=0.8, max_iter=10)
    label_spread.fit(latent_features, labels)
    return label_spread.transduction_, label_spread

def segment_customers(latent_features, n_clusters=3):
    """
    Step 3: KMeans.
    Clusters customers into Segments purely based on compressed behavioral traits.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    segments = kmeans.fit_predict(latent_features)
    return segments, kmeans

def map_recommendations(segments, interests):
    """
    Step 4: Recommendation System.
    Maps cluster outputs to targeted campaign types for Case 3 (Cooling Wipes).
    """
    campaign_map = {
        0: "Stay Ready Campaign: ไอเทมต้องพกก่อนเจอคนที่แคร์/เพื่อนร่วมงาน เพื่อรักษาความมั่นใจ",
        1: "Cool Down Campaign: ไอเทมดับร้อนระหว่างทำกิจกรรมกลางแจ้งหรือหลังออกกำลังกาย",
        2: "Instant Refresh Campaign: แคมเปญแจกสินค้าทดลอง (Sampling) สำหรับกลุ่มที่ยังไม่เคยใช้"
    }
    
    recommendations = []
    for seg, interest in zip(segments, interests):
        camp = campaign_map.get(seg, "General Outreach")
        
        if interest == 0:
            camp = "Awareness Push: " + camp
            
        recommendations.append({
            "Persona": int(seg), 
            "Intent_Topic": int(interest), 
            "Campaign": camp
        })
    return recommendations

def run_pipeline():
    features, labels = load_and_preprocess_data()
    print("Data Load: Ready.", features.shape[0], "customers found.")
    
    print("1. Autoencoder: Refining survey responses...")
    latent_feats, enc, scaler = train_autoencoder(features)
    
    print("2. SSL: Predicting missing customer intents...")
    interests, ssl = predict_interests_ssl(latent_feats, labels)
    
    print("3. KMeans: Grouping into actionable marketing Segments...")
    segments, kmeans = segment_customers(latent_feats)
    
    print("4. Mapping to Campaigns...")
    recs = map_recommendations(segments, interests)
    
    print("\n[SUCCESS] Pipeline complete. Sample assignment:")
    for i in range(min(5, len(recs))):
        print(f"Customer {i}: Persona {recs[i]['Persona']} => {recs[i]['Campaign']}")
    
    return (features, labels), (enc, scaler, ssl, kmeans)

if __name__ == "__main__":
    run_pipeline()
