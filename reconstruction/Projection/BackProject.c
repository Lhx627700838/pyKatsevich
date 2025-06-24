#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>


float * CalcLinePlaneIntersection(float normalVector[], float pointOnPlane[], float firstPointOnLine[], float seconPointOnLine[])
{
    float u[3], v[3];
    float sI;
    float N = 0, D = 0;
    int i;
    static float intersectionCoordinate[3];

    for ( i = 0; i < 3; i++ )
    {
        u[i] = firstPointOnLine[i] - pointOnPlane[i];
        v[i] = seconPointOnLine[i] - firstPointOnLine[i];
        N = N - normalVector[i] * u[i];
        D = D + normalVector[i] * v[i];
    }
    sI = N / D;

    for ( i = 0; i < 3; i++ )
    {
        intersectionCoordinate[i] = firstPointOnLine[i] + sI * v[i];
    }

    return intersectionCoordinate;
}


float CalcInterpolatedDetectorReading(float pointCoordinate[], float projectionData[], float xUnitVector[], float yUnitVector[], float detectorResolution[],  int detectorDimension[], float detectorCoordinateFirst[])
{
    // coordinate index of intersection point on detector
    float pointXCoordinate = 0.0, pointYCoordinate = 0.0;
    int xCoordFirst, xCoordSecond, yCoordFirst, yCoordSecond;
    float xDistanceFirst, xDistanceSecond, yDistanceFirst, yDistanceSecond;
    float interpolatedDetectorReading;

    for ( int i = 0; i < 3; i++ )
    {
        pointXCoordinate = pointXCoordinate + (pointCoordinate[i] - detectorCoordinateFirst[i]) * xUnitVector[i];
        pointYCoordinate = pointYCoordinate + (pointCoordinate[i] - detectorCoordinateFirst[i]) * yUnitVector[i];
    }
    pointXCoordinate = pointXCoordinate / detectorResolution[0];
    pointYCoordinate = pointYCoordinate / detectorResolution[1];

    // return nan if point is outside of detector panel
    if ((pointXCoordinate < -0.5) || (pointYCoordinate < -0.5) || (pointXCoordinate > (detectorDimension[0] - 0.5)) || (pointYCoordinate > (detectorDimension[1] - 0.5)) )
    {
        return NAN;
    }
    // handle cases where intersection is on the edge detectors and on detector pixel centers
    if (pointXCoordinate <= 0)
    {
        xCoordFirst = 0;
        xCoordSecond = 0;
        xDistanceFirst = 1;
        xDistanceSecond = 1;
    }
    else if (pointXCoordinate >= (detectorDimension[0] - 1))
    {
        xCoordFirst = detectorDimension[0] - 1;
        xCoordSecond = detectorDimension[0] - 1;
        xDistanceFirst = 1;
        xDistanceSecond = 1;
    }
    else if (pointXCoordinate == round(pointXCoordinate))
    {
        xCoordFirst = pointXCoordinate;
        xCoordSecond = pointXCoordinate;
        xDistanceFirst = 1;
        xDistanceSecond = 1;
    }
    else
    {
        xCoordFirst = floor(pointXCoordinate);
        xCoordSecond = ceil(pointXCoordinate);
        xDistanceFirst = pointXCoordinate - xCoordFirst;
        xDistanceSecond = xCoordSecond - pointXCoordinate;
    }

    if (pointYCoordinate <= 0)
    {
        yCoordFirst = 0;
        yCoordSecond = 0;
        yDistanceFirst = 1;
        yDistanceSecond = 1;
    }
    else if (pointYCoordinate >= (detectorDimension[1] - 1))
    {
        yCoordFirst = detectorDimension[1] - 1;
        yCoordSecond = detectorDimension[1] - 1;
        yDistanceFirst = 1;
        yDistanceSecond = 1;
    }
    else if (pointYCoordinate == round(pointYCoordinate))
    {
        yCoordFirst = pointYCoordinate;
        yCoordSecond = pointYCoordinate;
        yDistanceFirst = 1;
        yDistanceSecond = 1;
    }
    else
    {
        yCoordFirst = floor(pointYCoordinate);
        yCoordSecond = ceil(pointYCoordinate);
        yDistanceFirst = pointYCoordinate - yCoordFirst;
        yDistanceSecond = yCoordSecond - pointYCoordinate;
    }

    float distanceSum = (xDistanceFirst + xDistanceSecond + yDistanceFirst + yDistanceSecond) * 2;

    // compute interpolated detector reading
    interpolatedDetectorReading = (projectionData[xCoordFirst * detectorDimension[1] + yCoordFirst] * (xDistanceFirst + yDistanceFirst) +
                                  projectionData[xCoordFirst * detectorDimension[1] + yCoordSecond] * (xDistanceFirst + yDistanceSecond) +
                                  projectionData[xCoordSecond * detectorDimension[1] + yCoordFirst] * (xDistanceSecond + yDistanceFirst) +
                                  projectionData[xCoordSecond * detectorDimension[1] + yCoordSecond] * (xDistanceSecond + yDistanceSecond)) / distanceSum;
    

    return interpolatedDetectorReading;
}


void BackProjectFunc( int *imageDimension, float *imageResolution, float *imageCoordinate,
                        float *sourceCoordinate, float sourceSAD,
                        int *detectorDimension, float *detectorCoordinate, float *detectorResolution,
                        float *normalVectorToDetectorPlane, float *xUnitVector, float *yUnitVector,
                        float *betaVector, float dbeta,
                        float *projectionView,
                        float *backProjectedData
                        )
{
    float detectorCoordinateThis[3];
    float detectorCoordinateFirst[3];
    float imageCoordinateThis[3];
    float magni;
    float * intersectionCoordinate;
    float interpolatedDetectorReading;

    int iX, iY, iZ, iCoord;

    int detectorIndexFirstX = 0;
    int detectorIndexFirstY = detectorDimension[0] * detectorDimension[1] * 1;
    int detectorIndexFirstZ = detectorDimension[0] * detectorDimension[1] * 2;

    detectorCoordinateFirst[0] = detectorCoordinate[detectorIndexFirstX];
    detectorCoordinateFirst[1] = detectorCoordinate[detectorIndexFirstY];
    detectorCoordinateFirst[2] = detectorCoordinate[detectorIndexFirstZ];

    for ( iX = 0; iX < imageDimension[0]; iX++ )
    {
        for ( iY = 0; iY < imageDimension[1]; iY++ )
        {
            for ( iZ = 0; iZ < imageDimension[2]; iZ++ )
            {   
                clock_t start_time = clock();
                for ( iCoord = 0; iCoord < 3; iCoord++ )
                {
                    imageCoordinateThis[iCoord] = imageCoordinate[iZ + imageDimension[2] * (iY + imageDimension[1] * (iX + imageDimension[0] * iCoord))];
                }
                
                intersectionCoordinate = CalcLinePlaneIntersection(normalVectorToDetectorPlane, detectorCoordinateFirst, sourceCoordinate, imageCoordinateThis);
                magni = sourceSAD / (sourceSAD + imageCoordinateThis[1] * betaVector[1] - imageCoordinateThis[0] * betaVector[0]);
                interpolatedDetectorReading = CalcInterpolatedDetectorReading(intersectionCoordinate, projectionView, xUnitVector, yUnitVector, detectorResolution, detectorDimension, detectorCoordinateFirst);
                backProjectedData[(iX * imageDimension[1] + iY) * imageDimension[2] + iZ] = pow(magni, 2) * interpolatedDetectorReading * dbeta;
                clock_t end_time = clock();
                double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
                //printf("B程序运行时间: %f 秒\n", time_spent);
            }

        }
    }
}


