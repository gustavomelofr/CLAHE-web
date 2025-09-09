import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configuração da página (título e ícone que aparecem na aba do navegador)
st.set_page_config(page_title="CLAHE - Melhorador de Imagens", page_icon="🦷")

# --- Funções Auxiliares ---
def aplicar_clahe(imagem, clip_limit):
    """Aplica o CLAHE em uma imagem."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    imagem_melhorada = clahe.apply(imagem)
    return imagem_melhorada

def para_bytes(imagem):
    """Converte uma imagem OpenCV para bytes para o botão de download."""
    _, buffer = cv2.imencode(".png", imagem)
    return buffer.tobytes()

# --- Interface do Aplicativo ---

st.title("🔬 Ferramenta de Contraste CLAHE")
st.write("Faça o upload de uma radiografia ou imagem odontológica para melhorar o contraste e realçar detalhes.")

# 1. Upload do arquivo
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Lê a imagem enviada
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imagem_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.header("Ajuste o Contraste em Tempo Real")
    
    # 2. Slider interativo
    clip_limit = st.slider(
        "Força do Contraste (Clip Limit)",
        min_value=1.0, 
        max_value=15.0, 
        value=3.0,  # Valor inicial
        step=0.5
    )

    # 3. Aplica o CLAHE com o valor do slider
    imagem_melhorada = aplicar_clahe(imagem_original, clip_limit)

    # 4. Exibe a comparação
    col1, col2 = st.columns(2)
    with col1:
        st.image(imagem_original, caption="Original", use_column_width=True)
    with col2:
        st.image(imagem_melhorada, caption=f"CLAHE (Força: {clip_limit})", use_column_width=True)

    st.header("Download do Resultado")
    
    # 5. Botão de Download
    st.download_button(
        label="Baixar Imagem Melhorada",
        data=para_bytes(imagem_melhorada),
        file_name=f"imagem_melhorada_clahe_{clip_limit}.png",
        mime="image/png"
    )
