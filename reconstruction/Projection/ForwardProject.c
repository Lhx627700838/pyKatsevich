#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>


void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void selectionSort(float arr1[], float arr2[], float arr3[], int n)
{
    int i, j, minIndex;
    // One by one move boundary of unsorted subarray
    for (i = 0; i < n-1; i++)
    {
        minIndex = i;
        for (j = i+1; j < n; j++)
        {
            if (arr1[j] < arr1[minIndex])
            {
                minIndex = j;
            }
        }
        // Swap the found minimum element with the first element
        swap(&arr1[minIndex], &arr1[i]);
        swap(&arr2[minIndex], &arr2[i]);
        swap(&arr3[minIndex], &arr3[i]);
    }
}

void removeDuplicate(float allInterceptX[], float allInterceptY[], float allInterceptZ[], int *nAllIntercept)
{
    int i, j, k;
    for( i = 0; i < *nAllIntercept; i++ )
    {
        for( j = i+1; j < *nAllIntercept; j++)
        {
            // If any duplicate found
            if ( (allInterceptX[i] == allInterceptX[j]) && (allInterceptY[i] == allInterceptY[j]) && (allInterceptZ[i] == allInterceptZ[j]) )
            {
                // Delete the current duplicate element
                for( k = j; k < *nAllIntercept - 1; k++)
                {
                    allInterceptX[k] = allInterceptX[k + 1];
                    allInterceptY[k] = allInterceptY[k + 1];
                    allInterceptZ[k] = allInterceptZ[k + 1];
                }
                // Decrement size after removing duplicate element
                *nAllIntercept = *nAllIntercept - 1;
                // If shifting of elements occur then don't increment j
                j--;
            }
        }
    }
}

float CalcRayDistance(float startCoordinate[], float endCoordinate[])
{
    float sumSquared = 0;
    for ( int i = 0; i < 3; i++)
    {
        sumSquared = sumSquared + pow((endCoordinate[i] - startCoordinate[i]), 2);
    }

    return sqrt(sumSquared);
}


float RayTracingDistanceBased(int imageDimension[], float imageCoordinate[], float imageResolution[], float imageData[], float sourceCoordinate[], float detectorCoordinateThis[])
{
    int nXImage = imageDimension[0];
    int nYImage = imageDimension[1];
    int nZImage = imageDimension[2];

    //define a the plane coordinate of image 
    float xPlanes[nXImage + 1], yPlanes[nYImage + 1], zPlanes[nZImage + 1];
    for ( int iX = 0; iX <= nXImage; iX++ )
    {
        xPlanes[iX] = (iX - nXImage / 2.0) * imageResolution[0];
    }
    for ( int iY = 0; iY <= nYImage; iY++ )
    {
        yPlanes[iY] = (iY - nYImage / 2.0) * imageResolution[1];
    }
    for ( int iZ = 0; iZ <= nZImage; iZ++ )
    {
        zPlanes[iZ] = (iZ - nZImage / 2.0) * imageResolution[2];
    }

    //I can only understand this as a extrem situation that the xyz difference between s and d cover the whole image object 
    int nAllIntercept = 0;
    if (detectorCoordinateThis[0] != sourceCoordinate[0])
    {
        nAllIntercept = nAllIntercept + nXImage + 1;
    }
    if (detectorCoordinateThis[1] != sourceCoordinate[1])
    {
        nAllIntercept = nAllIntercept + nYImage + 1;
    }
    if (detectorCoordinateThis[2] != sourceCoordinate[2])
    {
        nAllIntercept = nAllIntercept + nZImage + 1;
    }

    // find intercept of source-detector line with x-, y- and z- planes
    float allInterceptX[nAllIntercept], allInterceptY[nAllIntercept], allInterceptZ[nAllIntercept];
    int interceptCounter = 0;
    if (detectorCoordinateThis[0] != sourceCoordinate[0])
    {
        for ( int iX = 0; iX <= nXImage; iX++ )
        {
            float gradVectorX = (xPlanes[iX] - sourceCoordinate[0]) / (detectorCoordinateThis[0] - sourceCoordinate[0]);
            allInterceptX[interceptCounter] = xPlanes[iX];
            allInterceptY[interceptCounter] = sourceCoordinate[1] + gradVectorX * (detectorCoordinateThis[1] - sourceCoordinate[1]);
            allInterceptZ[interceptCounter] = sourceCoordinate[2] + gradVectorX * (detectorCoordinateThis[2] - sourceCoordinate[2]);
            interceptCounter++;
        }
    }

    if (detectorCoordinateThis[1] != sourceCoordinate[1])
    {
        for ( int iY = 0; iY <= nYImage; iY++ )
        {
            float gradVectorY = (yPlanes[iY] - sourceCoordinate[1]) / (detectorCoordinateThis[1] - sourceCoordinate[1]);
            allInterceptX[interceptCounter] = sourceCoordinate[0] + gradVectorY * (detectorCoordinateThis[0] - sourceCoordinate[0]);
            allInterceptY[interceptCounter] = yPlanes[iY];
            allInterceptZ[interceptCounter] = sourceCoordinate[2] + gradVectorY * (detectorCoordinateThis[2] - sourceCoordinate[2]);
            interceptCounter++;
        }
    }

    if (detectorCoordinateThis[2] != sourceCoordinate[2])
    {
        for ( int iZ = 0; iZ <= nZImage; iZ++ )
        {
            float gradVectorZ = (zPlanes[iZ] - sourceCoordinate[2]) / (detectorCoordinateThis[2] - sourceCoordinate[2]);
            allInterceptX[interceptCounter] = sourceCoordinate[0] + gradVectorZ * (detectorCoordinateThis[0] - sourceCoordinate[0]);
            allInterceptY[interceptCounter] = sourceCoordinate[1] + gradVectorZ * (detectorCoordinateThis[1] - sourceCoordinate[1]);
            allInterceptZ[interceptCounter] = zPlanes[iZ];
            interceptCounter++;
        }
    }

    // Remove duplicate coordinates of all intercepts
    removeDuplicate(allInterceptX, allInterceptY, allInterceptZ, &nAllIntercept);

    // Sort the coordinates of all intercepts
    if (detectorCoordinateThis[0] != sourceCoordinate[0])
    {
        selectionSort(allInterceptX, allInterceptY, allInterceptZ, nAllIntercept);
    } else {
        selectionSort(allInterceptY, allInterceptX, allInterceptZ, nAllIntercept);
    }

    // imageWeight stores the length of the ray traversing through each voxel
    float projSum = 0.0;

    for ( int iIntercept = 0; iIntercept < (nAllIntercept-1); iIntercept++ )
    {
        float startCoordinate[3], endCoordinate[3], middleCoordinate[3];
        int xCoordVoxel, yCoordVoxel, zCoordVoxel;
        int imageIndexFirstX = 0;
        int imageIndexFirstY = imageDimension[0] * imageDimension[1] * imageDimension[2];
        int imageIndexFirstZ = imageDimension[0] * imageDimension[1] * imageDimension[2] * 2;
        float rayLength;
        
        startCoordinate[0] = allInterceptX[iIntercept];
        startCoordinate[1] = allInterceptY[iIntercept];
        startCoordinate[2] = allInterceptZ[iIntercept];

        endCoordinate[0] = allInterceptX[iIntercept+1];
        endCoordinate[1] = allInterceptY[iIntercept+1];
        endCoordinate[2] = allInterceptZ[iIntercept+1];

        middleCoordinate[0] = (startCoordinate[0] + endCoordinate[0]) / 2;
        middleCoordinate[1] = (startCoordinate[1] + endCoordinate[1]) / 2;
        middleCoordinate[2] = (startCoordinate[2] + endCoordinate[2]) / 2;

        xCoordVoxel = round((middleCoordinate[0] - imageCoordinate[imageIndexFirstX]) / imageResolution[0]);
        yCoordVoxel = round((middleCoordinate[1] - imageCoordinate[imageIndexFirstY]) / imageResolution[1]);
        zCoordVoxel = round((middleCoordinate[2] - imageCoordinate[imageIndexFirstZ]) / imageResolution[2]);

        if ( (xCoordVoxel >= 0) && (xCoordVoxel < nXImage) && (yCoordVoxel >= 0) && (yCoordVoxel < nYImage) && (zCoordVoxel >= 0) && (zCoordVoxel < nZImage))
        {
            rayLength = CalcRayDistance(startCoordinate, endCoordinate);
            projSum = projSum + rayLength * imageData[zCoordVoxel + imageDimension[2] * (yCoordVoxel + imageDimension[1] * xCoordVoxel)];
        }
    }

    return projSum;
}


void ForwardProjectFunc( int *imageDimension, float *imageResolution, float *imageCoordinate, float *imageData,
                        float *sourceCoordinate,
                        int *detectorDimension, float *detectorCoordinate,
                        float *forwardProjectedData)
{
    float detectorCoordinateThis[3];

    for ( int iX = 0; iX < detectorDimension[0]; iX++ )
    {
        for ( int iY = 0; iY < detectorDimension[1]; iY++ )
        {   
            //clock_t start_time = clock();
            for ( int iCoord = 0; iCoord < 3; iCoord++ )
            {
                detectorCoordinateThis[iCoord] = detectorCoordinate[iY + detectorDimension[1] * (iX + detectorDimension[0] * iCoord)];
            }
            forwardProjectedData[iX * detectorDimension[1] + iY] = RayTracingDistanceBased(imageDimension, imageCoordinate, imageResolution, imageData, sourceCoordinate, detectorCoordinateThis);
            //clock_t end_time = clock();
            //double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            //printf("F程序运行时间: %f 秒\n", time_spent);
        }
    }
}


