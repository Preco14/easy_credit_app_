import streamlit as st
import pandas as pd
import joblib
import os

# Cargar modelo
modelo_path = r"C:\Users\fprec\OneDrive\Documentos\MODELO_CREDITO\modelo_morosidad.pkl"
logo_path = r"C:\Users\fprec\OneDrive\Documentos\MODELO CREDITO\logo.jpg"

# Estilo
st.set_page_config(page_title="Easy Credit - Evaluador de Morosidad", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #4f78c7;
    }
    .title {
        font-size: 32px;
        color: white;
        text-align: center;
        margin-bottom: 15px;
    }
    .subtitle {
        font-size: 16px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo y título
if os.path.exists(logo_path):
    st.image(logo_path, width=100)

st.markdown('<div class="title">Easy Credit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Usamos Machine Learning para predecir la probabilidad de impago de un cliente y segmentar su perfil crediticio de forma inteligente y automatizada.</div>', unsafe_allow_html=True)

# Formulario
with st.form("formulario_cliente"):
    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input("Edad", min_value=0, max_value=100, step=1)
        zona = st.selectbox("Zona de vivienda", ["", "1", "2", "3", "4"])
        vivienda = st.selectbox("Vivienda", ["", "Propia", "Renta", "Familiar"])
        genero = st.selectbox("Género", ["", "Masculino", "Femenino", "Otro"])
        escolaridad = st.selectbox("Escolaridad", ["", "Primaria", "Secundaria", "Preparatoria", "Licenciatura", "Posgrado"])

    with col2:
        estado_civil = st.selectbox("Estado civil", ["", "Soltero", "Casado", "Unión libre", "Divorciado", "Viudo"])
        hijos = st.number_input("Número de hijos", min_value=0, max_value=15, step=1)
        ocupacion = st.selectbox("Ocupación", ["", "Asalariado", "Independiente"])
        ingresos = st.number_input("Ingresos mensuales", min_value=0.0, max_value=10000000.0, step=100.0, format="%.2f")
        score = st.number_input("Score crediticio", min_value=300, max_value=850, step=1)

    credito = st.selectbox("¿Tiene crédito vigente?", ["", "Sí", "No"])

    enviar = st.form_submit_button("Predecir morosidad")

    if enviar:
        campos = [zona, vivienda, genero, escolaridad, estado_civil, ocupacion, credito]
        if "" in campos:
            st.error("Por favor, completa todos los campos del formulario.")
        else:
            try:
                modelo = joblib.load(modelo_path)
                cliente = {
                    "edad": edad,
                    "zona_de_vivienda": zona,
                    "vivienda": vivienda,
                    "genero": genero,
                    "escolaridad": escolaridad,
                    "estado_civil": estado_civil,
                    "hijos": hijos,
                    "ocupacion": ocupacion,
                    "ingresos_mensuales": ingresos,
                    "score": score,
                    "credito_vigente": credito
                }
                df_cliente = pd.DataFrame([cliente])
                prob = modelo.predict_proba(df_cliente)[0][1]
                st.success(f"✅ Probabilidad de morosidad: **{prob:.2%}**")
            except Exception as e:
                st.error(f"❌ Error al predecir: {e}")
