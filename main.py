from regression_ui import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import random

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from PyQt5.QtWidgets import*
from PyQt5.QtGui import QPixmap
import math
import os
from Regressor import Regressor

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.pushButtonClasificar.clicked.connect(self.clasificarButton)
        self.pushButtonBrowse.clicked.connect(self.browseFile)
        self.spinBoxNumeroPatrones.valueChanged.connect(self.numeroPatronesValueChange)
        # self.spinBoxDimensionPatron.valueChanged.connect(self.numeroClasesValueChange)
        # self.spinBoxK.valueChanged.connect(self.kValueChange)

        # self.tabWidget.currentChanged.connect(self.tabChange)
        self.x_deafualt = [3,6,5,5,3,4,9,8,9,7,6,5,8,6]
        self.y_deafualt = [55,68,64,66,62,65,74,75,73,69,73,68,73,71]
        self.file_name = ""
        self.regressor = Regressor()


    def kValueChange(self):
        self.clasificador.setK(self.spinBoxK.value())

    # def tabChange(self):
    #     if self.tabWidget.currentIndex() == 1:
    #         datos = self.getDataTable()
    #         print('getDataTable()==>',datos)
    #         patron = self.getPatronDesconocido()
    #         self.clasificador.setPatronDesconcido(patron)
    #         self.clasificador.setPatrones(datos)
    #         self.rellenarDatosOrdenados()
    #         if self.spinBoxDimensionPatron.value() == 2:
    #             if (not datos is None) and (not patron is None):
    #                 self.generarGrafico(datos, patron, 'graficoTodos')
    #                 # loading image
    #                 self.pixmap = QPixmap('graficoTodos.png')
    #                 # adding image to label
    #                 self.imagenTodos.setPixmap(self.pixmap)
    #
    #                 self.generarGraficoK(datos, patron, self.clasificador.getPatronK().getDistancia())
    #                 # loading image
    #                 self.pixmap = QPixmap('graficoK.png')
    #                 # adding image to label
    #                 self.imagenK.setPixmap(self.pixmap)
    #
    #                 # self.clasificador.setClases(datos)
    #                 # self.generarGrafico(self.clasificador.getRepresentantes(), patron, 'graficoRepre')
    #                 # # loading image
    #                 # self.pixmap = QPixmap('graficoRepre.png')
    #                 # # adding image to label
    #                 # self.imagenRepresentantes.setPixmap(self.pixmap)
    #             else:
    #                 self.imagenTodos.setText('Datos erroneos')
    #                 self.imagenRepresentantes.setText('Datos erroneos')


    def numeroPatronesValueChange(self):
        #Se dan el numero de filas solicitadas
        self.tableDatos.setRowCount(self.spinBoxNumeroPatrones.value())
        #Se agregan los nombres a las filas
        rowsNames = []
        for i in range(self.spinBoxNumeroPatrones.value()):
            rowsNames.append(str(i + 1))

        self.tableDatos.setVerticalHeaderLabels(rowsNames)
        self.rellenarTabla()

    def rellenarTabla(self):
        for i in range(self.spinBoxNumeroPatrones.value()):#Rows
            for j in range(2):#Coulumns
                item = self.tableDatos.takeItem(i, j)
                if item is None:
                    self.tableDatos.setItem(i, j, QTableWidgetItem('0'))
                else:
                    self.tableDatos.setItem(i, j, item)

    def getDataTable(self):
        x = []
        y = []
        for i in range(self.spinBoxNumeroPatrones.value()):#Rows(Patrones)
            x_tmp = self.tableDatos.item(i, 0).text()
            y_tmp = self.tableDatos.item(i, 1).text()

            if self.isNumber(x_tmp):
                x.append(float(x_tmp))
            else:
                return None, None

            if self.isNumber(y_tmp):
                y.append(float(y_tmp))
            else:
                return None,None

        return x, y



    def isNumber(self, number):
        try:
            float(number)
            return True
        except ValueError:
            return False

    def generarGraficoData(self, x, y, nameFile='data'):
        plt.scatter(x, y, color='g')
        plt.title('Datos')
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.savefig(nameFile + '.png', dpi=75)
        plt.cla()
        plt.clf()

    def generarGraficoRegresion(self, x, y, y_pred, nameFile):
        plt.scatter(x, y, color='g')
        plt.plot(x,y_pred)
        plt.title(nameFile)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.savefig(nameFile + '.png', dpi=75)
        plt.cla()
        plt.clf()


    def browseFile(self):
        self.file_name = QFileDialog.getOpenFileName(self, 'Open file', '/home/lara/Desktop/Proba/Regression')[0]
        self.label_path.setText(self.file_name)
    def getDataFromFile(self):
        try:
            df = pd.read_csv(self.file_name)
            x, y = df['X'].tolist(), df['Y'].tolist()
            return x, y
        except:
            print('Error en el archivo, asegurese de tener una columna X y Y sin valores nulos')
            self.labelRes.setText('Error en el archivo, asegurese de tener una columna X y Y sin valores nulos')
            return None, None
    def clasificarButton(self):
        print('Regresion')
        if self.radioButtonManual.isChecked():
            x, y = self.getDataTable()
            if (not x is None) and (not y is None):
                print(x, y)
            else:
                self.labelRes.setText('Datos erroneos')
                return

        elif self.radioButtonFile.isChecked():
            if self.file_name != "":
                x, y = self.getDataFromFile()
                if not x is None:
                    print('From file:')
                    print(x, y)
            else:
                print('Ruta de archivo no válida')
                self.labelRes.setText('Ruta de archivo no válida')
                return

        else:
            x= self.x_deafualt
            y= self.y_deafualt
            print(self.x_deafualt, self.y_deafualt)


        #Plot data
        self.generarGraficoData(x,y, 'data')
        self.pixmap = QPixmap('data.png')
        self.imagenTodos.setPixmap(self.pixmap)

        #Lineal
        self.regressor.setValues(x,y)
        y_pred, mse = self.regressor.lineal()
        self.generarGraficoRegresion(x, y, y_pred, 'Regresión lineal')
        self.label_MSE_lineal.setText('MSE: ' + str(mse))
        self.pixmap = QPixmap('Regresión lineal.png')
        self.linealImage.setPixmap(self.pixmap)



    def __del__(self):
        pass
        #print('End exec')
        try:
            os.remove('graficoTodos.png')
            os.remove('graficoRepre.png')
        except:
            print('No se genero grafico')
        #os.remove('histVerduras.png')



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
