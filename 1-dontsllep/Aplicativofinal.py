## importaçao do opencv-python
import cv2

## criar uma variavel para camera
cap = cv2.VideoCapture(0)

## enquanto a cameraestiver aberta
while cap.isOpened():
    ## sucesso - booleana ( verificar se o frame esta vazio)
    ## frame - captura
    sucesso, frame = cap.read()

    ## realizar a verificaçao
    ## sucesso = 1   fracasso = 0
    if not sucesso:
        print("ignorando a frame vazio da camera")
        continue
    ##Carregar nosso frame - com titulo
    cv2.imshow('camera',frame)
    ## bitwise tabela ASC II
    ## 10 milissegundos
    ## ord() - retorna o valor Unicode (ou ASC II)
    ## o valor 0xFF é tabela ASC II estendida
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
cap.release()
cv2.destroyAllWindows()
    
