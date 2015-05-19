#include <iostream>
#include <random>
#include "mathutils.h"
#include "Vector3D.h"
#include "Transform.h"
using namespace std;

#define NUM_VERTICES_PER_BLADE 16

#pragma warning (disable: 4996)

float *vertices;
int *indices;
float *verticesPre;
float *verticesNex;


float *RestLength;
vec4 *Binormal;
vec4 *RefVector;
vec4 *GlobalFrames;



struct bladeVertex
{
	vec3 position;
	float padding;
    vec3 normal;
	float tex;
    vec3 tangent;
    float distance;
};

bladeVertex	bladeAsset[10][NUM_VERTICES_PER_BLADE];

bool LoadBladeAsset(const char* filename)
{
    FILE* fp = fopen( filename, "rb" );
    if( fp == NULL )
    {
		printf( "Error in reading file %s\n", filename );
        return false;
    }

    int lineno = 1;
    for( int i = 0; i < 10; ++i )
    {
        char line[1024];
        if( fgets( line, 1024, fp ) == NULL )
        {
            printf( "Error in reading line %d", lineno );
            return false;
        }

        const char* p = line;
        for( int j  = 0; j < NUM_VERTICES_PER_BLADE; ++j )
        {
            const char *end = strchr( p, ')' );
            char buffer[1024];
            strncpy( buffer, p, end - p + 1 );
            buffer[end - p + 1] = 0;
            float x, y, z;
            size_t n = sscanf( buffer, "(%f,%f,%f)", &x, &y, &z );
            bladeAsset[i][j].position = vec3( x, y, z );

            if( n != 3 )
            {
                printf( "Wrong format in line %d", lineno );
                return false;
            }
            p = end + 1;
        }

        lineno++;
    }

    fclose( fp );

    return true;
}




void GrowBlade( float x, float y, bladeVertex* blade, vec4& out_binormal,int c_count, int verticesNum)
{
	LoadBladeAsset("blade.txt");

	//Print bladeAsset data
	//for(int i = 0; i < 10; i++)
	//{
	//	for(int j = 0; j < 16; j++)
	//	{
	//		cout<<"(" <<bladeAsset[i][j].position[0]<< " "<< bladeAsset[i][j].position[0]
	//			<<" "<< bladeAsset[i][j].position[0]<<")"; 
	//	}
	//	cout<<endl;
	//}


    float position[2] = {x, y};

	bladeVertex* bladeKnots = (bladeVertex*)bladeAsset;

    // A random length of the blade
	float	lengthScaling = 1.0f;
    //lengthScaling *= m_bladeLength; // scale the blade length for the current scene.

    blade->position.x = position[0];
    blade->position.y = 0;
    blade->position.z = position[1];
    
	blade->tex = 0;
    blade->distance = 0;
    
    // Add random orientation to the blade
    /*float rotY = (float)rand() / (float)RAND_MAX;
    rotY = rotY * 2.0f - 1.0f;
    rotY *= 180.0f;*/

    float rotY = 90.0f;

    Matrix4x4 rotYMatrix;
    MatrixSetRotationY(rotYMatrix, rotY);
    
    // Lengthen the blade and reposition the blade to the seed position.
    for (int i = 1; i < verticesNum; ++i)
    {
        vec3 edge = bladeKnots[i].position - bladeKnots[i - 1].position;
        vec3 scaledEdge = edge * lengthScaling;
        vec3 rotatedEdge;
        Matrix4x4MultVec3( rotatedEdge, rotYMatrix, scaledEdge );

        blade[i].position = blade[i - 1].position + rotatedEdge;
        blade[i].tex = 0;

        //blade[i].distance += blade[i - 1].distance + rotatedEdge.Length();
    }

    // Compute the normal, tangents and other properties of each knot on the blade.
    vec4 out(0, 0, -1, 0);
    vec4 binormal;
    Matrix4x4MultVec4( binormal, rotYMatrix, out ); 
    vec3 b( binormal.x, binormal.y, binormal.z );

    out_binormal = b;

    for( int i = 0; i < NUM_VERTICES_PER_BLADE; ++i )
    {
        if (i == 0)
        {
            vec3 t = blade[1].position - blade[0].position;
            t.Normalize();
            vec3 n = t.Cross(b);
            n.Normalize();
            blade[0].normal = n;
            blade[0].tangent = t;
        }
        else if( i == NUM_VERTICES_PER_BLADE - 1 )
        {
            vec3 t = blade[NUM_VERTICES_PER_BLADE - 1].position - blade[NUM_VERTICES_PER_BLADE - 2].position;
            t.Normalize();
            vec3 n = t.Cross(b);
            n.Normalize();
            blade[NUM_VERTICES_PER_BLADE - 1].normal = n;
            blade[NUM_VERTICES_PER_BLADE - 1].tangent = t;
        }
        else
        {
            vec3 t0 = blade[i + 1].position - blade[i].position;
            vec3 t1 = blade[i].position - blade[i - 1].position;
            vec3 t = t0 + t1;
            t.Normalize();
            vec3 n = t.Cross(b);
            n.Normalize();
            blade[i].normal = n;
            blade[i].tangent = t;
        }

        if( i >= 1 )
        {
            vec3 d = blade[i - 1].position - blade[i].position;
            blade[i].distance = blade[i - 1].distance + d.Length();
        }
        else
        {
            blade[i].distance = 0.0f;
        }
    }

    for (int i = 1; i < NUM_VERTICES_PER_BLADE - 1; ++i)
    {
        blade[i].distance /= blade[NUM_VERTICES_PER_BLADE - 1].distance;
    }

    blade[NUM_VERTICES_PER_BLADE - 1].distance = 1.0f;
}

void importBladeAsset(int verticesNum)
{
}

void generateGrass(int verticesNum)
{
	importBladeAsset(verticesNum);

	int indicesNum = (verticesNum - 1) * 6;

	vertices = new float[verticesNum * 12];// pos (4), tangent(4), normaldistance(4)
    verticesPre = new float[verticesNum * 12];
	indices = new int[indicesNum];

    memset(vertices, 0, sizeof(float)* verticesNum *12);
    memset(verticesPre, 0, sizeof(float)* verticesNum *12);
    memset(indices, 0, sizeof(float)* verticesNum *12);

	RestLength = new float[verticesNum];
	Binormal = new vec4[1];
	RefVector = new vec4[verticesNum];
	GlobalFrames = new vec4[verticesNum];

    memset(RestLength, 0, sizeof(float) * verticesNum);
    memset(RefVector, 0, sizeof(float) * verticesNum);
    memset(GlobalFrames, 0, sizeof(float) * verticesNum);

	int *ind = indices;
	for(int j = 0; j < verticesNum-1; j++)
	{
		//  p2 --- p3 
        //  | \    |
        //  |   \  |
        //  p0 --- p1 
        
		ind[j * 6]     = j * 2;            // p0
        ind[j * 6 + 1] = (j + 1) * 2;      // p2 
        ind[j * 6 + 2] = j * 2 + 1;        // p1

        ind[j * 6 + 3] = (j + 1) * 2;      // p2
        ind[j * 6 + 4] = (j + 1) * 2 + 1;  // p3
        ind[j * 6 + 5] = j * 2 + 1;        // p1
	}

	vec4 bi;
	GrowBlade(0, 0, (bladeVertex*)vertices, bi, 0, verticesNum);
	*Binormal = bi;

	//print vertices data
	//for(int i = 0; i < 16; i++)
	//{
	//	for(int j = 0; j < 12; j++)
	//	{
	//		cout<<vertices[i*12 + j]<<" ";
	//	}
	//	cout<<endl;
	//}

	bladeVertex* blade = (bladeVertex*)vertices;

	float* l = RestLength;
	for (int j = 0; j < verticesNum - 1; ++j)
    {
		l[j] = (blade[j].position - blade[j+1].position).Length();
    }
	l[verticesNum - 1] = 0;


}

void constructFrames(int numVertices)
{
    // Local and global frames
    CTransform* globalTransform = new CTransform[numVertices];
    CTransform* localTransform = new CTransform[numVertices];

    for ( int j = 0; j < numVertices; ++j )
    {
        int idx = j;

        // vertex 0
        if ( j == 0 )
        {
			float* p1 = &vertices[idx * 3 * 4];
            float* p2 = &vertices[(idx + 1) * 3 * 4];

            CVector3D vert_i(p1[0], p1[1], p1[2]);
            CVector3D vert_i_plus_1(p2[0], p2[1], p2[2]);

            const CVector3D vec = vert_i_plus_1 - vert_i;
            CVector3D vecX = vec.NormalizeOther();

            CVector3D vecZ = vecX.Cross(CVector3D(1.0, 0, 0));

            if ( vecZ.LengthSqr() < 0.0001 )
            {
                vecZ = vecX.Cross(CVector3D(0, 1.0f, 0));
            }

            vecZ.Normalize();
            CVector3D vecY = vecZ.Cross(vecX).Normalize();

            CMatrix33 rotL2W;
            rotL2W(0, 0) = vecX.m_X;	rotL2W(0, 1) = vecY.m_X;		rotL2W(0, 2) = vecZ.m_X;
            rotL2W(1, 0) = vecX.m_Y;	rotL2W(1, 1) = vecY.m_Y;		rotL2W(1, 2) = vecZ.m_Y;
            rotL2W(2, 0) = vecX.m_Z;	rotL2W(2, 1) = vecY.m_Z;		rotL2W(2, 2) = vecZ.m_Z;

            localTransform[idx].GetRotation() = rotL2W;
            localTransform[idx].GetTranslation() = vert_i;
            globalTransform[idx] = localTransform[idx]; // For vertex 0, local and global transforms are the same. 
			RefVector[idx] = kiMath::vec4(0, 0, 0, 0);
        }
        else 
        {
			float* p0 = &vertices[(idx - 1) * 3 * 4];
            float* p1 = &vertices[idx * 3 * 4];

            CVector3D vert_i_minus_1(p0[0], p0[1], p0[2]);
            CVector3D vert_i(p1[0], p1[1], p1[2]);

            CVector3D vec = vert_i - vert_i_minus_1;
            vec = globalTransform[idx-1].GetRotation().InverseOther() * vec;

            CVector3D vecX = vec.NormalizeOther();

            CVector3D X = CVector3D(1.0f, 0, 0);
            CVector3D rotAxis = X.Cross(vecX);
            float angle = acos(X.Dot(vecX));

            if ( abs(angle) < 0.001 || rotAxis.LengthSqr() < 0.001 )
            {
                localTransform[idx].GetRotation().SetIdentity();
            }
            else
            {
                rotAxis.Normalize();
                CQuaternion rot = CQuaternion(rotAxis, angle);
                localTransform[idx].GetRotation() = rot;
            }

            localTransform[idx].GetTranslation() = vec;
            globalTransform[idx] = globalTransform[idx-1] * localTransform[idx];
			RefVector[idx] = kiMath::vec4(vec.m_X, vec.m_Y, vec.m_Z, 0);
        }

        CQuaternion q = globalTransform[idx].GetRotation();
        q.Normalize();
		GlobalFrames[idx] = kiMath::vec4(q.m_X, q.m_Y, q.m_Z, q.m_W);
    }

    delete [] globalTransform;
    delete [] localTransform;
}