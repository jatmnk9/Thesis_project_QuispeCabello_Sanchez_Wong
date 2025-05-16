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
from sklearn.metrics import classification_report

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
    # Tokenizar el texto (usando SpaCy o split() simple)
    #words = text.split()  # o [token.text for token in nlp(text)]
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_punct]
    total_words = len(words)
    
    # Evitar división por cero
    if total_words == 0:
        return pd.Series([0, 0, 0])
    
    # Contar palabras de cada emoción (como ya haces)
    anger_score = 0
    joy_score = 0
    sadness_score = 0
    
    for entry in lexicon_entries:
        phrase = entry['Text'].lower()
        pattern = re.escape(phrase)
        regex_pattern = r'\b' + pattern + r'\b'
        matches = len(re.findall(regex_pattern, text.lower()))
        
        # Sumar scores (aquí puedes ponderar por frecuencia o usar 1 por match)
        anger_score += matches * entry.get('Anger', 0)
        joy_score += matches * entry.get('Joy', 0)
        sadness_score += matches * entry.get('Sadness', 0)
    
    # Normalizar por longitud del texto (opcional: multiplicar por 100 para porcentaje)
    anger_norm = (anger_score / total_words) * 100
    joy_norm = (joy_score / total_words) * 100
    sadness_norm = (sadness_score / total_words) * 100
    
    return pd.Series([anger_norm, joy_norm, sadness_norm])

# Aplicar análisis avanzado
df_reviews[['anger', 'joy', 'sadness']] = df_reviews['review_clean'].progress_apply(analyze_sentiment_advanced)

# Preparar datos para modelos
X = df_reviews[['anger', 'joy', 'sadness']]
y = df_reviews['etiqueta'].str.strip().str.lower()

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Clasificación basada en emociones del léxico
def clasificar_por_emociones(row):
    joy = row['joy']
    neg = row['anger'] + row['sadness']
    
    if joy > neg:
        return 'positivo'
    elif neg > joy:
        return 'negativo'
    else:
        return 'neutro'

# Aplicar clasificación por emociones
df_reviews['prediccion_lexicon'] = df_reviews.apply(clasificar_por_emociones, axis=1)

# Comparar con etiquetas reales (convertir a minúsculas y limpiar)
df_reviews['etiqueta_limpia'] = df_reviews['etiqueta'].str.strip().str.lower()

# Calcular accuracy
aciertos = (df_reviews['prediccion_lexicon'] == df_reviews['etiqueta_limpia']).sum()
total = len(df_reviews)
accuracy = aciertos / total * 100

print(f"\n🔍 Resultados de clasificación por léxico:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Aciertos: {aciertos} de {total}")

print(classification_report(df_reviews['etiqueta_limpia'], df_reviews['prediccion_lexicon']))

# Guardar resultados en nuevo archivo
output_columns = ['review', 'review_clean', 'anger', 'joy', 'sadness', 'etiqueta', 'prediccion_lexicon']
df_reviews[output_columns].to_excel("Resultados_Lexicon.xlsx", index=False)

print("\n✅ Análisis por léxico completado. Revisa 'Resultados_Lexicon.xlsx'")