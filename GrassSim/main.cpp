#include <iostream>
#include "myFloat.h"
#include "Grass.h"
#include "simulation.h"
#include "draw.h"
using namespace std;

extern float vertices[128*12];
extern int indices[128*12];
extern float verticesPre[128];
extern float verticesNex[128];


extern float RestLength[128];
extern vec4 Binormal[128];
extern vec4 RefVector[128];
extern vec4 GlobalFrames[128];

extern vec4 sharedPos[128];
extern vec4 sharedLength[128];


int main()
{
	generateGrass(16);

	for(int i = 0; i < 16; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			cout<<vertices[i*12 + j]<<" ";
		}
		cout<<endl;
	}

	cout<<"-----------------------------------------------------------------------"<<endl;

	//void IntegrationAndGlobalShapeConstraints(int vertexInd);

	//void LocalShapeConstraints(int vertexInd);

	//void LengthConstriantsAndWind(int vertexInd);

	//void CollisionAndTangents(int vertexInd);

	for(int i = 0; i < 16; i++)	
	{
		IntegrationAndGlobalShapeConstraints(i);
		LocalShapeConstraints(i);
		LengthConstriantsAndWind(i);
		CollisionAndTangents(i);

	}

	for(int i = 0; i < 16; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			cout<<vertices[i*12 + j]<<" ";
		}
		cout<<endl;
	}


	//glutInit();
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize (200, 200);
    glutInitWindowPosition (10, 10);
    glutCreateWindow( "Point examples" );
    glutDisplayFunc( RenderScene );
    SetupRC();
    glutMainLoop();

	return 0;
}