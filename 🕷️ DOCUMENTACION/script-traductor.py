# Requisitos:
# pip install spacy deep-translator transformers torch
# python -m spacy download es_core_news_sm

import spacy
from deep_translator import GoogleTranslator
from transformers import MarianMTModel, MarianTokenizer

# Cargar modelo NER en español
nlp = spacy.load("es_core_news_sm")

# Diccionario técnico de ciberseguridad conceptual
terminos_cyber = {
    "malware": "malware",
    "ransomware": "ransomware",
    "vulnerabilidad": "vulnerability",
    "ataque de denegación de servicio": "denial-of-service attack",
    "ingeniería social": "social engineering",
    "firewall": "firewall",
    "registro de eventos": "event log",
    "penetración de red": "network penetration",
    # puedes ampliar
}

# Modelo de traducción (MarianMT)
model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def traducir_frase(texto):
    doc = nlp(texto)
    # Detectar conceptos y reemplazar por traducción técnica
    for ent in doc.ents:
        if ent.text.lower() in terminos_cyber:
            texto = texto.replace(ent.text, terminos_cyber[ent.text.lower()])

    # Traducir el resto de la frase con modelo
    translated = model.generate(**tokenizer(texto, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Ejemplo
texto = "El malware comprometió el firewall y provocó un ataque de denegación de servicio."
print(traducir_frase(texto))

