##pip install opencv-python

##Descargar el OPENCV y copiarlo en tu computadora
##https://medium.com/@sh.tsang/tutorial-opencv-v4-2-0-installation-in-windows-10-eca7c2c8c300

import cv2
import urllib.request
import numpy as np

class Imagenes:

    def __init__(self,url):
      self.url=url

    def identificarostro(self):
        resp = urllib.request.urlopen(self.url)
        imagen = np.asarray(bytearray(resp.read()), dtype="uint8")
        imagen = cv2.imdecode(imagen, cv2.IMREAD_COLOR)
        grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) ##no es necesario para funcionar pero puedes filtrar la imagen en escala de grises

        faceClassif = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = faceClassif.detectMultiScale(grises,
          scaleFactor=1.1, ##que tanto se reduce la imagen para saber cuantas imagenes van a procesar
          minNeighbors=5, ##cuantos rectangulos vecinos van a tener el rostro para no tener varios rectangulos para el mismo rostro
          minSize=(30,30), ##objetos menos a este tamaño serán ignorados
          maxSize=(200,200)) ##objetos mayores a este tamaño serán ignorados

        for (x,y,w,h) in faces:
          cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow('imagen',imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return imagen

url=input("Ingrese la url del archivo imagen jpg o png: ")

img=Imagenes(url)
img.identificarostro()