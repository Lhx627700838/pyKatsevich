/*=========================================================================
Copyright (c) 2015 - 2024 Zap Surgical Systems, Inc. All rights reserved.
=========================================================================*/

#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifndef _CUDADEVICES_
#define _CUDADEVICES_

#include <vector>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "CudaUtils.cu"

__device__ bool FindIntersectedPointsOnVolumeSurface(
	Point3D<float> &source, Point3D<float> &detectorPixel,
	Point3D<float> &origin, Vector3D<int> &VolumeSize,
	Vector3D<float> &VolumeRes, float *Points)
{
	
	//1. create a ray and 6 planes
	Plane pBottom;
	pBottom.Normal.Z = 1.f;
	pBottom.Distance = origin.Z-VolumeRes.Z/2.f;
	//		if (B.Z > pBottom.Distance)
	//			throw exception("Geometry error bottom surface of the bounding box");

	Plane pTop;
	pTop.Normal.Z = 1.f;
	pTop.Distance = pBottom.Distance + VolumeSize.Z*VolumeRes.Z;
	//		if (A.Z < pTop.Distance)
	//			throw exception("Geometry error top surface of the bounding box");

	Plane pLeft;
	pLeft.Normal.X = 1.f;
	pLeft.Distance = origin.X - VolumeRes.X / 2.f;

	Plane pRight;
	pRight.Normal.X = 1.f;
	pRight.Distance = pLeft.Distance + VolumeSize.X*VolumeRes.X;

	Plane pfront;
	pfront.Normal.Y = 1.f;
	pfront.Distance = origin.Y - VolumeRes.Y / 2.f;

	Plane pBack;
	pBack.Normal.Y = 1.f;
	pBack.Distance = pfront.Distance + VolumeSize.Y*VolumeRes.Y;

	Vector3D<float> bounding1(origin.X - VolumeRes.X / 2.f, origin.Y - VolumeRes.Y / 2.f, origin.Z - VolumeRes.Z / 2.f);

	Vector3D<float> bounding2(bounding1.X + VolumeSize.X*VolumeRes.X, bounding1.Y + VolumeSize.Y*VolumeRes.Y,
		bounding1.Z + VolumeSize.Z*VolumeRes.Z);
	//// 2. find intersection points of 6 planes
	Vector3D<float> Ptop;
	Vector3D<float> psource(source);
	Vector3D<float> pdetectorPixel(detectorPixel);
	bool bTop = pTop.IsIntersectLine(psource, pdetectorPixel, Ptop);
	Vector3D<float> Pbottom;
	bool bBottom = pBottom.IsIntersectLine(psource, pdetectorPixel, Pbottom);
	Vector3D<float> Pleft;
	bool bLeft = pLeft.IsIntersectLine(psource, pdetectorPixel, Pleft);
	Vector3D<float> Pright;
	bool bRight = pRight.IsIntersectLine(psource, pdetectorPixel, Pright);
	Vector3D<float> Pfront;
	bool bFront = pfront.IsIntersectLine(psource, pdetectorPixel, Pfront);
	Vector3D<float> Pback;
	bool bBack = pBack.IsIntersectLine(psource, pdetectorPixel, Pback);
	// 3. find 2 intersected points on 6 planes
	Vector3D<float> pa = Ptop;
	Vector3D<float> pb = Ptop;

	// pa is on top
	if (bTop && Ptop.X >= bounding1.X && Ptop.X <= bounding2.X && Ptop.Y >= bounding1.Y && Ptop.Y <= bounding2.Y)
	{
		pa = Ptop;
		if (bBottom && Pbottom.X >= bounding1.X && Pbottom.X <= bounding2.X && Pbottom.Y >= bounding1.Y && Pbottom.Y <= bounding2.Y)
			pb = Pbottom;
		else if (bLeft && Pleft.Y >= bounding1.Y && Pleft.Y <= bounding2.Y && Pleft.Z >= bounding1.Z && Pleft.Z <= bounding2.Z)
			pb = Pleft;
		else if (bRight && Pright.Y >= bounding1.Y && Pright.Y <= bounding2.Y && Pright.Z >= bounding1.Z && Pright.Z <= bounding2.Z)
			pb = Pright;
		else if (bFront && Pfront.X >= bounding1.X && Pfront.X <= bounding2.X && Pfront.Z >= bounding1.Z && Pfront.Z <= bounding2.Z)
			pb = Pfront;
		else if (bBack && Pback.X >= bounding1.X && Pback.X <= bounding2.X && Pback.Z >= bounding1.Z && Pback.Z <= bounding2.Z)
			pb = Pback;
	}
	// pa is on left
	else if (bLeft && Pleft.Y >= bounding1.Y && Pleft.Y <= bounding2.Y && Pleft.Z >= bounding1.Z && Pleft.Z <= bounding2.Z)
	{
		pa = Pleft;
		if (bBottom && Pbottom.X >= bounding1.X && Pbottom.X <= bounding2.X && Pbottom.Y >= bounding1.Y && Pbottom.Y <= bounding2.Y)
			pb = Pbottom;
		else if (bRight && Pright.Y >= bounding1.Y && Pright.Y <= bounding2.Y && Pright.Z >= bounding1.Z && Pright.Z <= bounding2.Z)
			pb = Pright;
		else if (bFront && Pfront.X >= bounding1.X && Pfront.X <= bounding2.X && Pfront.Z >= bounding1.Z && Pfront.Z <= bounding2.Z)
			pb = Pfront;
		else if (bBack && Pback.X >= bounding1.X && Pback.X<bounding2.X && Pback.Z >= bounding1.Z && Pback.Z <= bounding2.Z)
			pb = Pback;
	}
	// pa is on right
	else if (bRight && Pright.Y >= bounding1.Y && Pright.Y <= bounding2.Y && Pright.Z >= bounding1.Z && Pright.Z <= bounding2.Z)
	{
		pa = Pright;
		if (bBottom && Pbottom.X >= bounding1.X && Pbottom.X <= bounding2.X && Pbottom.Y >= bounding1.Y && Pbottom.Y <= bounding2.Y)
			pb = Pbottom;
		else if (bFront && Pfront.X >= bounding1.X && Pfront.X <= bounding2.X && Pfront.Z >= bounding1.Z && Pfront.Z <= bounding2.Z)
			pb = Pfront;
		else if (bBack && Pback.X >= bounding1.X && Pback.X <= bounding2.X && Pback.Z >= bounding1.Z && Pback.Z <= bounding2.Z)
			pb = Pback;
	}
	// pa is on bottom
	else if (bBottom && Pbottom.X >= bounding1.X && Pbottom.X <= bounding2.X && Pbottom.Y >= bounding1.Y && Pbottom.Y <= bounding2.Y)
	{
		pa = Pbottom;
		if (bFront && Pfront.X >= bounding1.X && Pfront.X <= bounding2.X && Pfront.Z >= bounding1.Z && Pfront.Z <= bounding2.Z)
			pb = Pfront;
		else if (bBack && Pback.X >= bounding1.X && Pback.X <= bounding2.X && Pback.Z >= bounding1.Z && Pback.Z <= bounding2.Z)
			pb = Pback;
	}
	else if (bFront && Pfront.X >= bounding1.X && Pfront.X <= bounding2.X && Pfront.Z >= bounding1.Z && Pfront.Z <= bounding2.Z)
	{
		pa = Pfront;
		if (bBack && Pback.X >= bounding1.X && Pback.X <= bounding2.X && Pback.Z >= bounding1.Z && Pback.Z <= bounding2.Z)
			pb = Pback;
	}
	if (pa != pb)
	{
		// find point sets between pa and pb step is stepRes
		Points[0] = pa.X;
		Points[1] = pa.Y;
		Points[2] = pa.Z;
		Points[3] = pb.X;
		Points[4] = pb.Y;
		Points[5] = pb.Z;
		return true;
	}
	return false;
}

__device__ float Dist(const Point3D<float> &left, const Point3D<float> &right)
{
	return sqrt((left.X - right.X)*(left.X - right.X) + (left.Y - right.Y)*(left.Y - right.Y)
		+ (left.Z - right.Z)*(left.Z - right.Z));
}

__device__ float cudaSoftwareInterpolatedPointInVolume(
	Point3D<float> &targetPoint, short *VolumeData,
	Point3D<float> &origin, Vector3D<int>&volumeSize,
	Vector3D<float> &spacing, float step)
{
	int &height = volumeSize.Y;
	int &width = volumeSize.X;
	int &depth = volumeSize.Z;
	if (height == 0 || width == 0 || depth == 0)
		return 0.f;

	Vector3D<float> bounding1(origin.X - spacing.X / 2.f, origin.Y - spacing.Y / 2.f, origin.Z - spacing.Z / 2.f);

	Vector3D<float> bounding2(bounding1.X + volumeSize.X*spacing.X, bounding1.Y + volumeSize.Y*spacing.Y,
		bounding1.Z + volumeSize.Z*spacing.Z);

	float tSum = 0.f;

	//if (bounding1.X <= targetPoint.X && targetPoint.X <= bounding2.X
	//	&& bounding1.Y <= targetPoint.Y && targetPoint.Y <= bounding2.Y
	//	&& bounding1.Z <= targetPoint.Z && targetPoint.Z <= bounding2.Z)  // it matches hardware interpolation of CUDA without this
	{
		int nx = (int)((targetPoint.X - bounding1.X) / spacing.X);
		int ny = (int)((targetPoint.Y - bounding1.Y) / spacing.Y);
		int nz = (int)((targetPoint.Z - bounding1.Z) / spacing.Z);

		if (nx >= width - 1)
			nx = width - 2;
		else if (nx < 0)
			nx = 0;
		if (ny >= height - 1)
			ny = height - 2;
		else if (ny < 0)
			ny = 0;
		if (nz >= depth - 1)
			nz = depth - 2;
		else if (nz<0)
			nz = 0;

		float target[3];
		target[0] = (targetPoint.X - (bounding1.X + nx*spacing.X)) / spacing.X;
		if (target[0] > 1.f)
			target[0] = 1.f;
		else if (target[0] < 0.f)
			target[0] = 0.f;
		target[1] = (targetPoint.Y - (bounding1.Y + ny*spacing.Y)) / spacing.Y;
		if (target[1] > 1.f)
			target[1] = 1.f;
		else if (target[1] < 0.f)
			target[1] = 0.f;
		target[2] = (targetPoint.Z - (bounding1.Z + nz*spacing.Z)) / spacing.Z;
		if (target[2] > 1.f)
			target[2] = 1.f;
		else if (target[2] < 0.f)
			target[2] = 0.f;
		int slice = width * height;

		short *Vbase = VolumeData + nz * slice + ny * width + nx;
		float Vvalue[8];

		Vvalue[0] = (float)* Vbase;
		Vvalue[1] = (float)*(Vbase + width);
		Vvalue[2] = (float)*(Vbase + width + 1);
		Vvalue[3] = (float)*(Vbase + 1);
		Vvalue[4] = (float)*(Vbase + slice);
		Vvalue[5] = (float)*(Vbase + slice + width);
		Vvalue[6] = (float)*(Vbase + slice + width + 1);
		Vvalue[7] = (float)*(Vbase + slice + 1);

		float t_interp = Trilinear(target, Vvalue);

		float aFactor = 0.1f * 0.000196f * (t_interp + 1024.0f);
		//float step = 1.0f;
		tSum = aFactor; // *step;
	}
	return tSum;
}

__device__ float cudaHardwareInterpolatedPointInVolume(
	cudaTextureObject_t texObject, Point3D<float> &targetPoint,
	Point3D<float> &origin, Vector3D<int>&volumeSize,
	Vector3D<float> &spacingInv)
{
	float fx = ((targetPoint.X - origin.X) * spacingInv.X) + 0.5f;
	float fy = ((targetPoint.Y - origin.Y) * spacingInv.Y) + 0.5f;
	float fz = ((targetPoint.Z - origin.Z) * spacingInv.Z) + 0.5f;
	float t_interp = tex3D<float>(texObject, fx, fy, fz) * SHRT_MAX;

	// TODO: need to cite source of this value
	const float linearAttenuationCoeff = 0.1f * 0.000196f;
	float aFactor = linearAttenuationCoeff * (t_interp + 1024.0f);
	return aFactor;
}

__device__ float  cudaFindAndSoftwareInterpolatedPointsOnARay(
	short *VolumeData, Point3D<float> &source,
	Point3D<float> &detector, Point3D<float> &origin,
	Vector3D<int>&VolumeSize, Vector3D<float> &VolumeRes, float step)
{
	float Points[6];
	float imgV = 0.f;

	if (FindIntersectedPointsOnVolumeSurface(source, detector, origin, VolumeSize, VolumeRes, Points))
	{
		Point3D<float> p1(Points[0], Points[1], Points[2]);
		Point3D<float> p2(Points[3], Points[4], Points[5]);
		// Find all points on the ray.
		Vector3D<float>v2v1(p2 - p1);
		Normalize1(v2v1);
		float dist = Dist(p1, p2);
		int n = (int)(dist / step);
		v2v1 = v2v1*step;
		Vector3D<float> tmpV(p1);
		imgV += cudaSoftwareInterpolatedPointInVolume(tmpV, VolumeData, origin, VolumeSize, VolumeRes, step);
		for (int i = 1; i < n - 1; i++)
		{
			tmpV += v2v1;
			imgV += cudaSoftwareInterpolatedPointInVolume(tmpV, VolumeData, origin, VolumeSize, VolumeRes, step);
		}
		tmpV = p2;
		imgV += cudaSoftwareInterpolatedPointInVolume(tmpV, VolumeData, origin, VolumeSize, VolumeRes, step);
	}
	return imgV;
}

__device__ float  cudaFindAndHardwareInterpolatedPointsOnARay(
	cudaTextureObject_t texObject, Point3D<float> &source,
	Point3D<float> &detector, Point3D<float> &origin,
	Vector3D<int>&VolumeSize, Vector3D<float> &VolumeRes)
{
	float Points[6];
	float imgV = 0.f;

	if (FindIntersectedPointsOnVolumeSurface(source, detector, origin, VolumeSize, VolumeRes, Points))
	{
		Point3D<float> p1(Points[0], Points[1], Points[2]);
		Point3D<float> p2(Points[3], Points[4], Points[5]);
		// Find all points on the ray.
		Vector3D<float>v2v1(p2 - p1);
		float dist = Normalize1(v2v1);
		int n = (int)dist;
		Vector3D<float> tmpV(p1);
		Vector3D<float> spacingInv;
		// precalculate inverses to remove inner-loop divides
		spacingInv.X = 1.0f / VolumeRes.X;
		spacingInv.Y = 1.0f / VolumeRes.Y;
		spacingInv.Z = 1.0f / VolumeRes.Z;
		v2v1.X = v2v1.X * spacingInv.X;
		v2v1.Y = v2v1.Y * spacingInv.Y;
		v2v1.Z = v2v1.Z * spacingInv.Z;
		imgV += cudaHardwareInterpolatedPointInVolume(texObject, tmpV, origin, VolumeSize, spacingInv);
		float fx = ((tmpV.X - origin.X) * spacingInv.X) + 0.5f;
		float fy = ((tmpV.Y - origin.Y) * spacingInv.Y) + 0.5f;
		float fz = ((tmpV.Z - origin.Z) * spacingInv.Z) + 0.5f;
		// TODO: need to cite source of this value
		const float linearAttenuationCoeff = 0.1f * 0.000196f;
        //TODO: try float4 for vector add
		for (int i = 1; i < n - 1; i++)
		{
			fx += v2v1.X;
			fy += v2v1.Y;
			fz += v2v1.Z;
			float t_interp = tex3D<float>(texObject, fx, fy, fz) * SHRT_MAX;
			float aFactor = linearAttenuationCoeff * (t_interp + 1024.0f);
			imgV += aFactor;
		}
		tmpV = p2;
		imgV += cudaHardwareInterpolatedPointInVolume(texObject, tmpV, origin, VolumeSize, spacingInv);
	}
	return imgV;
}

__global__  void cudaCaculateDDRImageSoftwareInterp(DRRGenParams *params)
{
	const  int index = blockIdx.y*blockDim.x + threadIdx.x;
	Point3D<float> source(params->SourceX, params->SourceY, params->SourceZ);
	Point3D<float> detector((float)threadIdx.x + 0.5f, (float)blockIdx.y + 0.5f, 0.f);
	__shared__ float shared_trans[9];  // 3X3 matrix
	if (threadIdx.x < 9)
	{
		shared_trans[threadIdx.x] = params->transMatrix[threadIdx.x];
	}
	__syncthreads();
	Point3D<float> translation(params->translationX, params->translationY, params->translationZ);
	detector = shared_trans * detector + translation;

	Point3D<float> origin(params->OriginX, params->OriginY, params->OriginZ);
	Vector3D<float> VolumeRes(params->VolumeResX, params->VolumeResY, params->VolumeResZ);
	Vector3D<int> VolumeSize(params->VolumeSizeX, params->VolumeSizeY, params->VolumeSizeZ);

	float integrationValue = cudaFindAndSoftwareInterpolatedPointsOnARay(params->VolumeData, source, detector, origin, VolumeSize, VolumeRes, 1.f);
   	params->DDRImgOutput[index] = integrationValue;
}

__global__  void  cudaCaculateDDRImageHardwareInterp(DRRGenParams *params, cudaTextureObject_t texObject)
{
	const  int index = blockIdx.y*blockDim.x + threadIdx.x;
	Point3D<float> source(params->SourceX, params->SourceY, params->SourceZ);
	// XXX "incestuous DRR loopback test" suggests that we should NOT have this half-pixel offset here XXX
	// XXX but until Mohan speaks with Qing leaving it in XXX
	Point3D<float> detector((float)threadIdx.x + 0.5f, (float)blockIdx.y + 0.5f, 0.f);
	__shared__ float shared_trans[9];  // 3X3 matrix
	if (threadIdx.x < 9)
	{
		shared_trans[threadIdx.x] = params->transMatrix[threadIdx.x];
	}
	__syncthreads();
	Point3D<float> translation(params->translationX, params->translationY, params->translationZ);
	detector = shared_trans*detector + translation;

	Point3D<float> origin(params->OriginX, params->OriginY, params->OriginZ);
	Vector3D<float> VolumeRes(params->VolumeResX, params->VolumeResY, params->VolumeResZ);
	Vector3D<int> VolumeSize(params->VolumeSizeX, params->VolumeSizeY, params->VolumeSizeZ);

	float integrationValue = cudaFindAndHardwareInterpolatedPointsOnARay(texObject, source, detector, origin, VolumeSize, VolumeRes);
   	params->DDRImgOutput[index] = integrationValue;
}

#endif