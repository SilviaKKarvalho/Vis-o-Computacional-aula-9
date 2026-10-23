

# import cv2
# from ultralytics import YOLO

# def detectar_pessoas(caminho_imagem, caminho_saida="resultado_deteccao.jpg", conf_minima=0.5):
#     """
#     Detecta pessoas em uma imagem utilizando YOLOv8 em CPU.
    
#     Args:
#         caminho_imagem (str): Caminho para a imagem de entrada.
#         caminho_saida (str): Caminho para salvar a imagem processada.
#         conf_minima (float): Limiar de confiança para filtragem (0 a 1).
#     """
#     # 1. Carrega o modelo YOLOv8 pré-treinado (versão nano para CPU)
#     model = YOLO("yolov8n.pt")
    
#     # 2. Carrega a imagem com o OpenCV
#     imagem = cv2.imread(caminho_imagem)
#     if imagem is None:
#         raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_imagem}")
    
#     # 3. Executa a inferência forçando o uso da CPU
#     resultados = model(imagem, conf=conf_minima, device="cpu")
    
#     # 4. Processa os resultados e desenha as marcações na imagem
#     for resultado in resultados:
#         boxes = resultado.boxes
#         for box in boxes:
#             # Identifica a classe detectada (0 no COCO = 'person')
#             id_classe = int(box.cls[0])
#             nome_classe = model.names[id_classe]
            
#             if nome_classe == "person":
#                 # Extrai as coordenadas da caixa delimitadora
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 confianca = float(box.conf[0])
                
#                 # Desenha o retângulo ao redor da pessoa
#                 cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
#                 # Adiciona o rótulo com a taxa de confiança
#                 texto = f"Pessoa: {confianca:.2f}"
#                 cv2.putText(imagem, texto, (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
#     # 5. Salva e exibe o resultado
#     cv2.imwrite(caminho_saida, imagem)
#     print('Imagem processada com sucesso e salva em: {caminho_saida}')

# # Exemplo de uso:
# detectar_pessoas("p.png")




# import cv2
# from ultralytics import YOLO

# def iniciar_deteccao_webcam():
#     """
#     Abre a webcam padrão do computador e realiza detecção de objetos/pessoas 
#     em tempo real utilizando YOLOv8 em CPU.
#     """
#     # 1. Carrega o modelo YOLOv8 pré-treinado (versão nano para otimizar a CPU)
#     print("Carregando o modelo YOLOv8...")
#     model = YOLO("yolov8n.pt")
    
#     # 2. Inicializa a webcam (0 geralmente indica a câmera padrão)
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Erro: Não foi possível acessar a webcam.")
#         return

#     # Opcional: Ajustar resolução para melhorar a fluidez em CPU (ex: 640x480)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     print("Iniciando transmissão. Pressione 'q' para sair.")

#     while cap.isOpened():
#         sucesso, frame = cap.read()
#         if not sucesso:
#             print("Erro ao capturar o frame da câmera.")
#             break

#         # 3. Executa a inferência no frame atual (forçando CPU)
#         # verbose=False oculta logs repetitivos no terminal a cada frame
#         resultados = model(frame, device="cpu", verbose=False)
        
#         # 4. Desenha as detecções (bounding boxes, rótulos e confiança) no frame
#         frame_anotado = resultados[0].plot()

#         # 5. Exibe o vídeo em uma janela do OpenCV
#         cv2.imshow("Deteccao em Tempo Real - YOLOv8 (CPU)", frame_anotado)

#         # Interrompe o loop se a tecla 'q' for pressionada
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 6. Libera os recursos da câmera e fecha as janelas
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Encerrado com sucesso.")

# if __name__ == "__main__":
#     iniciar_deteccao_webcam()




import cv2
import numpy as np
import pyautogui
import streamlit as st
from ultralytics import YOLO

# Configurações de segurança do PyAutoGUI (jogue o mouse para o canto da tela para travar)
pyautogui.FAILSAFE = True

st.set_page_config(
    page_title="Controle de Mouse por Visão Computacional", page_icon="🖱️", layout="centered"
)

st.title("🖱️ Controle do Mouse com Movimento das Mãos (YOLOv8)")
st.write(
    "Utilize a sua mão detectada pela webcam para controlar o cursor do mouse na tela."
)

# Barra lateral com controles
st.sidebar.header("Configurações")
confianca = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.5, 0.05)
suavizacao = st.sidebar.slider("Fator de Suavização do Mouse", 1, 10, 3)

# Botões de controle de estado no Streamlit
if "executando" not in st.session_state:
    st.session_state.executando = False


col1, col2 = st.sidebar.columns(2)
if col1:
    if st.button("Iniciar Câmera"):
        st.session_state.executando = True
if col2:
    if st.button("Parar Câmera"):
        st.session_state.executando = False

# Espaço para exibir o vídeo na interface do Streamlit
quadro_video = st.empty()

if st.session_state.executando:
    # Carrega o modelo YOLOv8 Pose (baixado automaticamente na primeira execução)
    @st.cache_resource
    def carregar_modelo():
        return YOLO("yolov8n-pose.pt")


    model = carregar_modelo()

    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()

    # Variáveis para suavização de movimento
    prev_x, prev_y = 0, 0

    st.sidebar.success(
        "Câmera ativa! Mova a mão para controlar o mouse. (Pressione 'Parar Câmera' para encerrar)."
    )

    while st.session_state.executando:
        ret, frame = cap.read()
        if not ret:
            st.error("Não foi possível acessar a webcam.")
            break

        # Espelha a imagem para uma experiência mais natural (tipo espelho)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Executa a inferência YOLOv8 Pose na CPU
        resultados = model(frame, conf=confianca, device="cpu", verbose=False)

        # Processa os keypoints (pontos articulares)
        for r in resultados:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                # Pegamos o primeiro indivíduo detectado
                keypoints = r.keypoints.xy[0].cpu().numpy()

                # No YOLOv8-pose, os índices 9 e 10 correspondem aos pulsos (direito e esquerdo)
                # Vamos usar o pulso direito (índice 9) se disponível, senão o esquerdo (índice 10)
                pulso_x, pulso_y = 0, 0
                if len(keypoints) > 10 and (
                    keypoints[9][0] > 0 or keypoints[9][1] > 0
                ):
                    pulso_x, pulso_y = keypoints[9]
                elif len(keypoints) > 10 and (
                    keypoints[10][0] > 0 or keypoints[10][1] > 0
                ):
                    pulso_x, pulso_y = keypoints[10]

                if pulso_x > 0 and pulso_y > 0:
                    # Mapeia as coordenadas da câmera para a resolução da tela do computador
                    target_x = np.interp(pulso_x, [100, w - 100], [0, screen_w])
                    target_y = np.interp(pulso_y, [100, h - 100], [0, screen_h])

                    # Aplica suavização linear para evitar movimentos bruscos/trêmulos
                    curr_x = prev_x + (target_x - prev_x) / suavizacao
                    curr_y = prev_y + (target_y - prev_y) / suavizacao

                    # Move o mouse via PyAutoGUI
                    try:
                        pyautogui.moveTo(curr_x, curr_y, duration=0.0)
                    except Exception:
                        pass

                    prev_x, prev_y = curr_x, curr_y

                    # Desenha um círculo visual no pulso rastreado
                    cv2.circle(
                        frame,
                        (int(pulso_x), int(pulso_y)),
                        10,
                        (0, 255, 0),
                        -1,
                    )

        # Converte a imagem de BGR para RGB para exibição correta no Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        quadro_video.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True,
            caption="Rastreamento Ativo",
        )

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info(
        "Clique no botão **'Iniciar Câmera'** na barra lateral para começar a interagir."
    )