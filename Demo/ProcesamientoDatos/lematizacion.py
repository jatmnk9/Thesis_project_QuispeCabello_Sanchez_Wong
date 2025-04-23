import spacy
import unidecode
import pandas as pd
from tqdm import tqdm

# Cargar modelo spaCy
nlp = spacy.load("es_core_news_sm")

# Verbos que se deben eliminar
verbos_prohibidos = {"ser", "estar", "haber"}

# Lista de pronombres personales y posesivos (en minúscula, lematizados)
pronombres_personales = {
    "yo", "tu", "usted", "el", "ella", "nosotros", "nosotras", "vosotros", "vosotras", "ellos", "ellas",
    "me", "te", "se", "nos", "os", "lo", "la", "los", "las", "le", "les"
}
posesivos = {
    "mi", "mio", "mia", "mios", "mias",
    "tu", "tuyo", "tuya", "tuyos", "tuyas",
    "su", "suyo", "suya", "suyos", "suyas",
    "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras"
}

# Quitar tildes
def quitar_tildes(texto):
    return unidecode.unidecode(texto) if isinstance(texto, str) else ""

# Procesar texto
def procesar_review(review):
    if not isinstance(review, str):
        return ""

    texto = quitar_tildes(review)
    doc = nlp(texto)
    lemas = []

    for token in doc:
        if token.pos_ == "DET" and "PronType=Art" in token.morph:
            continue
        if token.lemma_.lower() in verbos_prohibidos and token.pos_ in {"VERB", "AUX"}:
            continue
        lemas.append(token.lemma_.lower())

    # Eliminación post-lematización de pronombres personales y posesivos
    lemas_filtrados = [lemma for lemma in lemas if lemma not in pronombres_personales and lemma not in posesivos]

    return " ".join(lemas_filtrados)

# Leer el Excel
df = pd.read_excel("aseguradoras_reviews.xlsx")

# Aplicar función con barra de progreso
tqdm.pandas(desc="Procesando reviews")
df["review_lemmatizada"] = df["review"].progress_apply(procesar_review)

# Guardar en Excel
df.to_excel("aseguradoras_reviews_procesado.xlsx", index=False)
print("✅ Listo. Reviews procesadas y guardadas.")
