import re
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def convertir_fecha(fecha_texto):
    """
    Convierte fechas relativas como "Hace un mes", "Hace 4 meses", "Hace un año",
    "una semana atrás", "2 semanas atrás" al formato mm/yy.
    """
    hoy = datetime.today()
    
    # Diccionario para manejar casos de "un" o "una"
    texto_a_numeral = {"un": 1, "una": 1}
    
    # Expresión regular para capturar tanto "Hace X tiempo" como "X tiempo atrás"
    match = re.search(r"(?:Hace|)(?:\s+|)(un|una|\d+)\s+(día|semana|mes|año)s?\s*(atrás)?", fecha_texto, re.IGNORECASE)
    
    if match:
        cantidad_texto = match.group(1).lower()  # Extrae "un", "una" o el número
        unidad = match.group(2).lower()  # Extrae "día", "semana", "mes" o "año"
        
        # Convierte "un" o "una" a 1, y si es número, lo convierte a entero
        cantidad = texto_a_numeral.get(cantidad_texto, int(cantidad_texto) if cantidad_texto.isdigit() else 1)
        
        if "día" in unidad:
            fecha_final = hoy - timedelta(days=cantidad)
        elif "semana" in unidad:
            fecha_final = hoy - timedelta(weeks=cantidad)
        elif "mes" in unidad:
            fecha_final = hoy - relativedelta(months=cantidad)
        elif "año" in unidad:
            fecha_final = hoy - relativedelta(years=cantidad)
        
        return fecha_final.strftime("%m/%y")
    
    return fecha_texto  # Si no se pudo convertir, devolver el texto original

# Cargar el archivo Excel
df = pd.read_excel("ministerios_reviews.xlsx")

# Aplicar la función a la columna review_date
df["review_date"] = df["review_date"].astype(str).apply(convertir_fecha)

# Guardar el archivo corregido
df.to_excel("ministerios_reviews_corregido.xlsx", index=False)