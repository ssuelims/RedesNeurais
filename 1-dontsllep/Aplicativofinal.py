## importaçao do opencv-python
import cv2
import mediapipe as mp

## criar uma variavel para camera
cap = cv2.VideoCapture(0) # um construtor

## Usando uma soluçao para desenhar pontos
mp_drawing = mp.solutions.drawing_utils

## usando uma soluçao paraFace Mesh Detection
mp_face_mesh = mp.solutions.face_mesh

## enquanto a camera estiver aberta

## Liberaçao automatica
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():
        ## sucesso - booleana ( verificar se o frame esta vazio)
        ## frame - captura
        sucesso, frame = cap.read()

        ## realizar a verificaçao
        ## sucesso = 1   fracasso = 0
        if not sucesso:
            print("ignorando a frame vazio da camera")
            continue

        ## Transformando de BGR para RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        ## Criar uma varivel
        saida_facemesh = facemesh.process(frame)
        ## o OpenCV - entende BGR
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB_RGB2BGR)
        """
        1- Mostrar os pontos da nossa face
        2- O process - processar os dados
        3- face_landmars(COORDENADAS)
        """
        for face_landmars in saida_facemesh.multi_face_landmars:
            # desenhar
            mp_drawing.draw_landmars(frame,face_landmars,mp_face_mesh.FACEMESH_CONTOURS)



        ## Carregar nosso frame - com titulo
    
        cv2.imshow('camera',frame)
        ## bitwise tabela ASC II
        ## 10 milissegundos 
        ## ord() - retorna o valor Unicode (ou ASC II)
        ## o valor 0xFF é tabela ASC II estendida
        ## c - 
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break
cap.release() # encera todo o processo da camera
cv2.destroyAllWindows() # fecha todas as
    
