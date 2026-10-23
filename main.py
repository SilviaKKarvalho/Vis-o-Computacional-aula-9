import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO, YOLOWorld

ALVOS = ["cell phone", "car", "house"]
NOMES_PT = {"cell phone": "Celular", "car": "Carro", "house": "Casa"}
CORES = {"cell phone": (255, 120, 0), "car": (0, 200, 0), "house": (0, 120, 255)}  # RGB
MAX_LADO = 640


@st.cache_resource(show_spinner="Carregando modelo YOLO...")
def carregar_modelo():
    """YOLO-World (vocabulário aberto) reconhece 'house'. Fallback: YOLOv8n COCO (sem 'house')."""
    try:
        modelo = YOLOWorld("yolov8s-worldv2.pt")
        modelo.set_classes(ALVOS)
        return modelo, True
    except Exception:
        return YOLO("yolov8n.pt"), False


def detectar(img_bgr, modelo, conf):
    h, w = img_bgr.shape[:2]
    escala = MAX_LADO / max(h, w)
    if escala < 1:  # reduz a imagem para economizar memória
        img_bgr = cv2.resize(img_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)

    res = modelo.predict(img_bgr, imgsz=MAX_LADO, conf=conf, device="cpu", verbose=False)[0]
    contagem = {k: 0 for k in ALVOS}

    for b in res.boxes:
        nome = res.names[int(b.cls)]
        if nome not in ALVOS:
            continue
        contagem[nome] += 1
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cor = CORES[nome][::-1]  # RGB -> BGR
        rotulo = f"{NOMES_PT[nome]} {float(b.conf):.0%}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), cor, 2)
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), cor, -1)
        cv2.putText(img_bgr, rotulo, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    saida = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    del res, img_bgr
    return saida, contagem


st.set_page_config(page_title="Detector de Objetos", page_icon="🔍", layout="centered")
st.title("🔍 Detector: Celulares, Carros e Casas")

modelo, tem_house = carregar_modelo()
if not tem_house:
    st.warning("YOLO-World indisponível: usando YOLOv8n (COCO). A classe 'Casa' NÃO será detectada.")

conf = st.slider("Confiança mínima", 0.05, 0.90, 0.25, 0.05)
arquivo = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if arquivo:
    img = cv2.imdecode(np.frombuffer(arquivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Não foi possível ler a imagem.")
    else:
        with st.spinner("Detectando..."):
            resultado, contagem = detectar(img, modelo, conf)
        del img

        st.image(resultado, use_container_width=True)

        cols = st.columns(3)
        for col, nome in zip(cols, ALVOS):
            col.metric(NOMES_PT[nome], contagem[nome])

        encontrados = [NOMES_PT[k] for k, v in contagem.items() if v > 0]
        if encontrados:
            st.success("A imagem contém: " + ", ".join(encontrados))
        else:
            st.info("Nenhum objeto-alvo encontrado.")