import pandas as pd
import re
import unicodedata
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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
df_lexicon = pd.read_excel("Lexicon.xlsx", sheet_name="lexiconv2_lematizado")

# Preprocesar léxico y reseñas
df_reviews['review_clean'] = df_reviews['review'].progress_apply(limpiar_texto)
df_lexicon['Text'] = df_lexicon['Text'].astype(str).str.lower()
df_lexicon['pattern'] = df_lexicon['Text'].apply(lambda x: re.escape(x) if ' ' in x else r'\b' + re.escape(x) + r'\b')
lexicon_entries = df_lexicon.to_dict(orient='records')

def analyze_sentiment_advanced(text):
    matched_phrases = []
    text_lower = text.lower()
    text_mask = ['0'] * len(text_lower)
    
    # Variables para emociones
    anger_score = 0
    joy_score = 0
    sadness_score = 0
    
    lexicon_sorted = sorted(lexicon_entries, key=lambda x: len(x['Text'].split()), reverse=True)

    for entry in lexicon_sorted:
        phrase = entry['Text'].lower()
        pattern = re.escape(phrase)
        regex_pattern = r'\b' + pattern + r'\b'
        
        for match in re.finditer(regex_pattern, text_lower):
            start, end = match.span()
            if all(c == '0' for c in text_mask[start:end]):
                matched_phrases.append(entry['Text'])
                # Sumar scores de emociones
                anger_score += entry.get('Anger', 0)
                joy_score += entry.get('Joy', 0)
                sadness_score += entry.get('Sadness', 0)
                for i in range(start, end):
                    text_mask[i] = '1'

    return pd.Series([anger_score, joy_score, sadness_score])

# Aplicar análisis avanzado
df_reviews[['anger', 'joy', 'sadness']] = df_reviews['review_clean'].progress_apply(analyze_sentiment_advanced)

# Preparar datos para modelos
X = df_reviews[['anger', 'joy', 'sadness']]
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
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name, 
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Imprimir reporte de clasificación detallado
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
    except Exception as e:
        print(f"Error con {name}: {str(e)}")
        continue

# Resultados
df_results = pd.DataFrame(results)
print("\nResultados de clasificación:")
print(df_results.sort_values('F1-Score', ascending=False))

# Exportar resultados completos
df_reviews.to_excel("REVISAR_AVANZADO.xlsx", index=False)
print("\n✅ Análisis avanzado completado. Revisa 'REVISAR_AVANZADO.xlsx'")