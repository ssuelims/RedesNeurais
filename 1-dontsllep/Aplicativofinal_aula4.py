#Objetivo da Aula 4(05/10/2024)

# Entender as coordenadas do Mediapipe
# Entender a normalizaçao e desnormalizaçao do ponto do MediaPipe
# Analizar os olhos (Seguindo o Artigo)
# Utilizar a funçao EAR(distancia Euclidiana) Seguindo o artigo
import cv2
import mediapipe as mp # importação do mediapipe para usar o facemesh
import numpy as np # FUNCAO EAR

#importação do opencv-python
# Pontos dos olhos
# olho esquerdo
p_olho_esq=[385,380,387,373,362,263]

# olho direito
p_olho_dir = [160,144,158,33,133]
#FIXME: inversao Euclidiana( para dia 06/11/2024)
# Fuçao EAR
def calculo_ear(face,p_olho_dir,p_olho_esq):
    #FIXME:list comprehension
    #[[[]]] Lista dentro de outra lista
    try:
        face = np.array([[coord.x,coord.y]]for coord in face)
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear



#criar uma variável para camera
cap = cv2.VideoCapture(0)
# usando uma solução de desenho
mp_drawing = mp.solutions.drawing_utils
# usando uma solução para o Face Mesh Detection
mp_face_mesh = mp.solutions.face_mesh
#liberação automática
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    # enquanto a camera estiver aberta
    while cap.isOpened():
        # sucesso-booleana (verificar se o frame esta vazio)
        # frame - captura
        sucesso, frame = cap.read()
        # realizar a verificação
        # sucesso = 1   fracasso = 0
        if not sucesso:
            print("ignorando o frame vazio da camêra")
            continue
        # transformando de BGR para RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # criar uma variável      dados processados (ex.pontos do rosto)
        # FIXME: Processamento do frame(saide_facemesh e o frame processado)
        saida_facemesh = facemesh.process(frame)
        # O OpenCV - entende BGR
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        # O try - tratando o erro de ausência de usuário em frente a camera
        try: 
            #mostrar os pontos, mostrar a detecção que o MediaPipe fez
            # vou criar uma variável face_landmarks - que são as coordenadas da nossa face
            # Ele vai ser atribuido ao nosso conjunto de coordenadas
            # saida_facemesh é o nosso conjunto de coordenadas
            # o multi_face_landmarks vai retornas as coordenadas
            # após isso ele deve desenhar esses pontos no nosso rosto
            #FIXME: acesso as coordenadas
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                # desenhar
                # uso o método draw_landmarks para pontuar os desenhos
                # dentro dos parenteses
                # o nosso (frame)
                # as coordenadas - (face_landmarks)
                # Especificar os nossos pontos : FACEMESH_CONTOURS
                mp_drawing.draw_landmarks(frame,
                                            face_landmarks,
                                            mp_face_mesh.FACEMESH_CONTOURS,
                                            landmarks_drawing_spec = mp_drawing.DrawingSpec(color=(255,102,102),thickness=1,circle_radius=1),
                                            Connection_drawing_spec= mp_drawing.DrawingSpec(color=(102,204,0),thickness=1,circle_radius=1)
                                        )
            #FIXME: normalizacao para pixel
        except:
            print("algo deu errado")
        finally:
            print("encerrando o processoo")    
        # carregar nosso frame - com título
        cv2.imshow('Camera',frame)
        # Estude sobre bitwise
        # Estude sobre tabela ASC II estendida
        # ord() - retorna o valor Unicode (ou ASC II) 
        # o valor 0xFF é tabela ASC II estendida
       
        if cv2.waitKey(10) & 0xFF in [ord('c'), ord('C')]:
            break
# Libera o recurso de captura de vídeo 
cap.release()
# Esse método fecha todas as janelas abertas pelo OpenCV.
cv2.destroyAllWindows()


#  O que e o ponto?
# Que coordenadas corresponde a tal ponto?
# Ids dos Pontos sao
# O..........................................467