import pandas as pd
import re
import unicodedata
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Modelos de clasificación (ejemplos)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Preparación
tqdm.pandas()
nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))

# Cargar modelo SpaCy
nlp = spacy.load("es_core_news_sm")

# Modificadores y verbos auxiliares
modificadores = {'no', 'nunca', 'jamás', 'nada', 'nadie', 'ninguno', 'ni', 'poco', 'muy', 'tan', 'bastante', 'demasiado', 'otro', 'mucho'}
verbos_auxiliares = {'ser', 'estar', 'haber'}

# Función para quitar tildes
def quitar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# Función de limpieza y lematización (manteniendo la original)
def limpiar_texto(texto):
    texto = quitar_tildes(str(texto).lower())
    doc = nlp(texto)
    tokens = [
        token.lemma_ 
        for token in doc 
        if token.is_alpha and (token.lemma_ not in stopwords_es or token.lemma_ in modificadores)
        and token.lemma_ not in verbos_auxiliares
    ]
    return ' '.join(tokens)

# Cargar datos
df_reviews = pd.read_excel("reviews_test.xlsx")
df_nrc = pd.read_excel("demo_lexicon/nrc.xlsx", sheet_name="nrc")

# Preprocesar NRC y reseñas
df_reviews['review_clean'] = df_reviews['review'].progress_apply(limpiar_texto)
df_nrc['Text'] = df_nrc['Text'].astype(str).str.lower()
df_nrc['pattern'] = df_nrc['Text'].apply(lambda x: re.escape(x) if ' ' in x else r'\b' + re.escape(x) + r'\b')
nrc_entries = df_nrc.to_dict(orient='records')

def analyze_sentiment_nrc(text):
    matched_phrases = []
    text_lower = text.lower()
    text_mask = ['0'] * len(text_lower)
    
    # Variables para emociones NRC
    positive_score = 0
    negative_score = 0
    anger_score = 0
    anticipation_score = 0
    disgust_score = 0
    fear_score = 0
    joy_score = 0
    sadness_score = 0
    surprise_score = 0
    trust_score = 0
    
    nrc_sorted = sorted(nrc_entries, key=lambda x: len(x['Text'].split()), reverse=True)

    for entry in nrc_sorted:
        phrase = entry['Text'].lower()
        pattern = re.escape(phrase)
        regex_pattern = r'\b' + pattern + r'\b'
        
        for match in re.finditer(regex_pattern, text_lower):
            start, end = match.span()
            if all(c == '0' for c in text_mask[start:end]):
                matched_phrases.append(entry['Text'])
                # Sumar scores de emociones NRC
                positive_score += entry.get('Positive', 0)
                negative_score += entry.get('Negative', 0)
                anger_score += entry.get('Anger', 0)
                anticipation_score += entry.get('Anticipation', 0)
                disgust_score += entry.get('Disgust', 0)
                fear_score += entry.get('Fear', 0)
                joy_score += entry.get('Joy', 0)
                sadness_score += entry.get('Sadness', 0)
                surprise_score += entry.get('Surprise', 0)
                trust_score += entry.get('Trust', 0)
                for i in range(start, end):
                    text_mask[i] = '1'

    return pd.Series([
        positive_score, negative_score, anger_score, anticipation_score,
        disgust_score, fear_score, joy_score, sadness_score,
        surprise_score, trust_score
    ])

# Aplicar análisis con NRC
df_reviews[['positive', 'negative', 'anger', 'anticipation', 
            'disgust', 'fear', 'joy', 'sadness', 
            'surprise', 'trust']] = df_reviews['review_clean'].progress_apply(analyze_sentiment_nrc)

# Preparar datos para modelos
X = df_reviews[['positive', 'negative', 'anger', 'anticipation', 
                'disgust', 'fear', 'joy', 'sadness', 
                'surprise', 'trust']]
y = df_reviews['etiqueta'].str.strip().str.lower()

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Diccionario de modelos
models = {
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Evaluar modelos
results = []
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': accuracy})
    except Exception as e:
        print(f"Error con {name}: {str(e)}")
        continue

# Resultados
df_results = pd.DataFrame(results)
print("\nResultados de clasificación:")
print(df_results.sort_values('Accuracy', ascending=False))

# Exportar resultados completos
df_reviews.to_excel("REVISAR_NRC.xlsx", index=False)
print("\n✅ Análisis con NRC completado. Revisa 'REVISAR_NRC.xlsx'")