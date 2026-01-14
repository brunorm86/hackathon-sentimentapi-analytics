import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from contextlib import asynccontextmanager

# --- Carregamento dos Modelos e Variaveis Globais ---
MODEL_PATH_MULTI = "./data/models/modelo_multi.joblib"
model_multi = None

# --- CONTRATO DE DADOS ---
class SentimentRequest(BaseModel):
    texto: str

# --- LIFESPAN (Substitui o @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lógica de STARTUP (Roda ao ligar)
    global model_multi

        # Carregar Modelo MULTI
    if os.path.exists(MODEL_PATH_MULTI):
        try:
            model_multi = joblib.load(MODEL_PATH_MULTI)
            print("✅ Modelo MULTI carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar Modelo MULTI: {e}")
    else:
        print(f"⚠️ Aviso: '{MODEL_PATH_MULTI}' não encontrado.")

    yield  # A aplicação roda aqui

    # Lógica de SHUTDOWN (Opcional: roda ao desligar)
    print("🛑 Desligando API e limpando recursos...")
    model_multi = None

# --- Configuração ---
# Agora a variável 'lifespan' existe, então essa linha funciona:
app = FastAPI(title="API de Predição ML", lifespan=lifespan)

# --- Rotas ---
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_multi_loaded": model_multi is not None
    }

@app.post("/predict")
def predict(request: SentimentRequest):

    if model_multi:
        try:
            pred = model_multi.predict([request.texto])[0]
            sent_pred = pred[0]
            lang_pred = pred[1]

            try:
                probs_list = model_multi.predict_proba([request.texto])
                probs_sent = probs_list[0][0]
                probs_lang = probs_list[1][0]

                moc = model_multi.named_steps['multioutputclassifier']
                classes_sent = moc.estimators_[0].classes_
                classes_lang = moc.estimators_[1].classes_

                idx_sent = list(classes_sent).index(sent_pred)
                idx_lang = list(classes_lang).index(lang_pred)

                proba_sent = float(probs_sent[idx_sent])
                proba_lang = float(probs_lang[idx_lang])
            except AttributeError:
                proba_sent = 1.0
                proba_lang = 1.0

            return {
                "sentimento": str(sent_pred),
                "prob_sentimento": f"{proba_sent * 100:.2f}%",
                "idioma": str(lang_pred),
                "prob_idioma": f"{proba_lang * 100:.2f}%"
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Caso caia aqui e o model_multi não esteja carregado
        return {"erro": "Modelo Multi não carregado no servidor"}

if __name__ == "__main__":
    uvicorn.run("app_v2:app", host="0.0.0.0", port=8000)