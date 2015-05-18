#include <iostream>
#include "myFloat.h"
#include "Grass.h"
using namespace std;

void printVector(float4 a)
{
	cout<<"("<<a.x<<","<<a.y<<","<<a.z<<","<<a.w<<")";
}

void printVector(float3 a)
{
	cout<<"("<<a.x<<","<<a.y<<","<<a.z<<")";
}

void printVector(float2 a)
{
	cout<<"("<<a.x<<","<<a.y<<")";
}

int main()
{
	generateGrass(16);

	return 0;
}