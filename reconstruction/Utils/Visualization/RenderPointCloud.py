"""
Modified Python 3 VTK script to display 3D
"""

import sys
import vtk
import numpy as np


class VtkPointCloud:
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e9):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(3)

    def addPoint(self, point):
        if (self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints):
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = np.random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


def LoadData(filename, pointCloud):
    data = np.genfromtxt(filename, dtype=float, usecols=[0, 1, 2])
    for iPoint in range(np.size(data, 0)):
        point = data[iPoint]
        #point = 20*(random.rand(3)-0.5)
        pointCloud.addPoint(point)

    return pointCloud


def RenderPointCloud(pointCloud):

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("System Visualization")
    renderWindowInteractor.Start()


def RenderPointCloudFromArray(data):

    pointCloud = VtkPointCloud()
    nPoint = np.shape(data)[-1]
    for iPoint in range(nPoint):
        point = [data[0, iPoint], data[1, iPoint], data[2, iPoint]]
        pointCloud.addPoint(point)

    RenderPointCloud(pointCloud)

if __name__ == '__main__':

    pointCloud = VtkPointCloud()
    pointCloud = LoadData(sys.argv[1], pointCloud)

    RenderPointCloud(pointCloud)
