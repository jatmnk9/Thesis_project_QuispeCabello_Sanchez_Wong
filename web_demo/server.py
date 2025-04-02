from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/api/ranking_historico', methods=['GET'])
def ranking_historico():
    ruta_excel = "sentiment_analysis_results.xlsx"  # Asegúrate de tener este archivo en la misma carpeta
    df = pd.read_excel(ruta_excel, sheet_name="Ranking_Historico")
    
    # Convertimos el DataFrame a una lista de diccionarios
    data = df.to_dict(orient="records")
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
