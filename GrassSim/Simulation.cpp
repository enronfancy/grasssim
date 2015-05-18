#include "Grass.h"
//#include "myFloat.h"
#include "mathutils.h"
#include <math.h>

#define FACTOR 2
#define SCENE_SIZE_FACTOR 1

#define THREAD_GROUP_SIZE 64
#define NUM_OF_ROCKS 256
#define NUM_VERTS_PER_STRAND 16
#define NUM_STRANDS_PER_GROUP (THREAD_GROUP_SIZE / NUM_VERTS_PER_STRAND)
#define NUM_VERTS_PER_TILE (SCENE_SIZE_FACTOR*4096/(FACTOR*FACTOR))

#define MAX_COLLISION_BALLS 64

extern float *vertices;
extern float *verticesPre;
extern int *indices;

vec4* vp  = (vec4*)verticesPre;
vec4* v = (vec4*)vertices;

vec4 sharedPos[128];

vec4 wind = vec4(0.0f, 0.0f, 0.0f, 0.0f);
vec4 gravity = vec4(0.0f, 0.0f, 0.0f, 0.0f);

int numELCIter = 2;
int numLSCIter = 3;
int bCollision;    
float timestep = 1.0f/60.0f;
float elpasedTime = 20.0f;    
float damping = 0.08f;
float padding0;
float padding1; 

int bladeOffset = 0;
float ballLeftTime = 0.0f;
float stiffnessLSC = 0.7f;
float padding2;
int ballSize = 2;
int balls[16];


bool IsMovable(vec4 particle)
{
    if ( particle.w > 0 )
        return true;
    return false;      
}

vec2 ConstraintMultiplier(vec4 particle0, vec4 particle1)
{
    if (IsMovable(particle0)) 
    {
        if (IsMovable(particle1))
            return vec2(0.5, 0.5);
        else
            return vec2(1, 0);
    }
    else 
    {
        if (IsMovable(particle1))
            return vec2(0, 1);
        else
            return vec2(0, 0);
    }    
}

vec4 MakeQuaternion(float angle_radian, vec3 axis)
{
    // create quaternion using angle and rotation axis
    vec4 quaternion;
    float halfAngle = 0.5f * angle_radian;
    float sinHalf = sin(halfAngle);

    quaternion.w = cos(halfAngle);
    quaternion.x = sinHalf * axis.x; 
    quaternion.y = sinHalf * axis.y; 
	quaternion.z = sinHalf * axis.z; 

    return quaternion;
}

vec4 InverseQuaternion(vec4 q)
{
    float lengthSqr = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;

    if ( lengthSqr < 0.001 )
        return vec4(0, 0, 0, 1.0f);

    q.x = -q.x / lengthSqr;
    q.y = -q.y / lengthSqr;
    q.z = -q.z / lengthSqr;
    q.w = q.w / lengthSqr;

    return q;
}


vec3 MultQuaternionAndVector(vec4 q, vec3 v)
{
    vec3 uv, uuv;
    vec3 qvec = vec3(q.x, q.y, q.z);
	uv = qvec.Cross(v);
    //uv = cross(qvec, v);
    //uuv = cross(qvec, uv);
	uuv = qvec.Cross(uv);
    uv *= (2.0f * q.w);
    uuv *= 2.0f;

    return v + uv + uuv;
}

vec4 MultQuaternionAndQuaternion(vec4 qA, vec4 qB)
{
    vec4 q;

    q.w = qA.w * qB.w - qA.x * qB.x - qA.y * qB.y - qA.z * qB.z;
    q.x = qA.w * qB.x + qA.x * qB.w + qA.y * qB.z - qA.z * qB.y;
    q.y = qA.w * qB.y + qA.y * qB.w + qA.z * qB.x - qA.x * qB.z;
    q.z = qA.w * qB.z + qA.z * qB.w + qA.x * qB.y - qA.y * qB.x;
    
    return q;
}

vec4 NormalizeQuaternion(vec4& q)
{
    float n = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;

    if ( n == 0 ) 
    {
        q.w = 1.f;
        return q;
    }

    n = 1.0f/sqrt(n);
    
    q.w *= n;
    q.x *= n;
    q.y *= n;
    q.z *= n;

    return q;
}

void ApplyDistanceConstraint(vec4& pos0, vec4& pos1, float targetDistance, float stiffness = 1.0)
{
    vec3 delta = vec3(pos1.x - pos0.x, pos1.y - pos0.y, pos1.z - pos0.z);
	float distance = std::max(delta.Length(), 1e-7f);
    float stretching = 1 - targetDistance / distance;
    delta = delta * stretching;
    vec2 multiplier = ConstraintMultiplier(pos0, pos1);
    
	//pos0.xyz += multiplier[0] * delta * stiffness;
	//pos1.xyz -= multiplier[1] * delta * stiffness;

	pos0.x += multiplier[0] * delta.x * stiffness;
	pos0.y += multiplier[0] * delta.y * stiffness;
	pos0.z += multiplier[0] * delta.z * stiffness;

	pos1.x += multiplier[1] * delta.x * stiffness;
	pos1.y += multiplier[1] * delta.y * stiffness;
	pos1.z += multiplier[1] * delta.z * stiffness;
}

void CalcIndicesInVertexLevel(int local_id, int group_id, int &globalStrandIndex, int &localStrandIndex, int &globalVertexIndex, int &localVertexIndex, int &numVerticesInTheStrand, int &indexForSharedMem)
{
    indexForSharedMem = local_id;
    int numOfStrandsPerThreadGroup = NUM_STRANDS_PER_GROUP; // Hard-coded
    numVerticesInTheStrand = NUM_VERTS_PER_STRAND; // again. hard-coded
 
    localStrandIndex = local_id % numOfStrandsPerThreadGroup;
    globalStrandIndex = group_id * numOfStrandsPerThreadGroup + localStrandIndex;
    localVertexIndex = (local_id - localStrandIndex) / numOfStrandsPerThreadGroup;  
    globalVertexIndex = globalStrandIndex * numVerticesInTheStrand + localVertexIndex;
}

vec4 Integrate(vec4 curPosition, vec4 oldPosition, vec4 force, int globalVertexIndex, int localVertexIndex, int numVerticesInTheStrand, float dampingCoeff = 1.0f)
{  
    vec4 outputPos = curPosition;

    //force.xyz += gravity.xyz;
    outputPos.x = curPosition.x + (1.0 - dampingCoeff)*(curPosition.x - oldPosition.x) + force.x*timestep*timestep; 
    outputPos.y = curPosition.y + (1.0 - dampingCoeff)*(curPosition.y - oldPosition.y) + force.y*timestep*timestep; 
    outputPos.z = curPosition.z + (1.0 - dampingCoeff)*(curPosition.z - oldPosition.z) + force.z*timestep*timestep; 
    return outputPos;  
}

struct CollisionCapsule
{
	vec4 p1; // xyz = position 1 of capsule, w = radius
	vec4 p2; // xyz = position 2 of capsule, w = radius * radius
};

//--------------------------------------------------------------------------------------
// 
//	CapsuleCollision
//
//  Moves the position based on collision with capsule
//
//--------------------------------------------------------------------------------------
vec3 CapsuleCollision(vec4 curPosition, vec4 oldPosition, CollisionCapsule cc, float friction = 0.4f)
{
    vec3 newPos = vec3(curPosition.x,curPosition.y,curPosition.z);
    const float radius = cc.p1.w;
    const float radius2 = cc.p2.w;
        
    if ( !IsMovable(curPosition) )
        return newPos;
            
    vec3 segment = vec3(cc.p2.x - cc.p1.x, cc.p2.y - cc.p1.y, cc.p2.z - cc.p1.z);
    vec3 delta1 = vec3(curPosition.x - cc.p1.x,curPosition.y - cc.p1.y,curPosition.z - cc.p1.z);
    vec3 delta2 = vec3(cc.p2.x - curPosition.x,cc.p2.y - curPosition.y,cc.p2.z - curPosition.z);
        
    float dist1 = dot(delta1, segment);
    float dist2 = dot(delta2, segment);
        
    // colliding with sphere 1
    if ( dist1 < 0.f )
    {
        if ( dot(delta1, delta1) < radius2 )
        {
            vec3 n = normalize(delta1);
            newPos = n*radius  + vec3(cc.p1.x,cc.p1.y, cc.p1.z);
        }
        return newPos;
    }
        
    // colliding with sphere 2
    if ( dist2 < 0.f )
    {
        if ( dot(delta2, delta2) < radius2 )
        {
            vec3 n = normalize(-delta2);
            newPos = n*radius  + vec3(cc.p2.x,cc.p2.y, cc.p2.z);
        }
        return newPos;
    }
        
    // colliding with middle cylinder
    vec3 x = (cc.p2._xyz()*dist1  + cc.p1._xyz()*dist2) / (dist1 + dist2);
    vec3 delta = curPosition._xyz() - x;
        
    if ( dot(delta, delta) < radius2 )
    {
        vec3 n = normalize(delta);
        vec3 vec = curPosition._xyz() - oldPosition._xyz();
        vec3 segN = normalize(segment);
        vec3 vecTangent = segN * dot(vec, segN) ;
        vec3 vecNormal = vec - vecTangent;
        newPos.x = oldPosition.x + vecTangent.x*friction + (vecNormal.x + n.x*radius - delta.x);
        newPos.y = oldPosition.y + vecTangent.y*friction + (vecNormal.y + n.y*radius - delta.y);
        newPos.z = oldPosition.z + vecTangent.z*friction + (vecNormal.z + n.z*radius - delta.z);
    }
    return newPos;
}

vec3 SphereCollision(vec4 curPosition, vec4 sphere)
{
    vec3 newPos = curPosition._xyz();
            
    if ( !IsMovable(curPosition) )
        return newPos;
    
    const vec3 center = sphere._xyz();
    const float radius = sphere.w;
        
    vec3 d = newPos - center;

    if ( dot(d, d) < radius*radius )
    {
        vec3 n = normalize(d);
        newPos = n* radius  + center;
        return newPos;
    }

    return newPos;
}

//--------------------------------------------------------------------------------------
// 
//  UpdateFinalVertexPositions
//
//  Updates the  hair vertex positions based on the physics simulation
//
//--------------------------------------------------------------------------------------
void UpdateFinalVertexPositions(vec4 oldPosition, vec4 newPosition, int globalVertexIndex, int localVertexIndex, int numVerticesInTheStrand)
{ 
    if ( localVertexIndex < numVerticesInTheStrand )
    {
        vp[globalVertexIndex] = oldPosition;
        v[globalVertexIndex * 3] = newPosition;
    }        
}

//--------------------------------------------------------------------------------------
// 
//  IntegrationAndGlobalShapeConstraints
//
//  Compute shader to simulate the gravitational force with integration and to maintain the
//  global shape constraints.
//
// One thread computes one vertex.
//
//--------------------------------------------------------------------------------------
void IntegrationAndGlobalShapeConstraints(int vertexInd)
{
    //uint globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem;
    //CalcIndicesInVertexLevel(GIndex, GId.x, globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem);
    
    vec4 currentPos = vec4(0, 0, 0, 0); // position when this step starts. In other words, a position from the last step. 
    
    // Copy data into shared memory 
    //currentPos = sharedPos[vertexInd] = g_HairVertices[vertexInd * 3];
    currentPos = sharedPos[vertexInd] = v[vertexInd * 3];

    //GroupMemoryBarrierWithGroupSync();

    if ( vertexInd == 0 || localVertexIndex == 1 )
    {
        sharedPos[vertexInd].w = 0;
        currentPos.w = 0;
    }
    else
    {
        sharedPos[vertexInd].w = 1.0f;
        currentPos.w = 1.0f;
    }

    // Integrate
    float dampingCoeff = damping;

    vec4 oldPos = vp[vertexInd];
    vec4 force = vec4(0, 0, 0, 0);

    if ( IsMovable(currentPos) )  
        sharedPos[vertexInd] = Integrate(currentPos, oldPos, force, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, dampingCoeff); 
    
    // update global position buffers
    UpdateFinalVertexPositions(currentPos, sharedPos[indexForSharedMem], globalVertexIndex, localVertexIndex, numVerticesInTheStrand);
}
