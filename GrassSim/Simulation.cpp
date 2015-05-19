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

static vec4 sp = vec4(0,0,-2, 0.25);

extern float *vertices;
extern float *verticesPre;
extern int *indices;



extern float *RestLength;
extern vec4 *Binormal;
extern vec4 *RefVector;
extern vec4 *GlobalFrames;

vec4 sharedPos[128];
vec4 sharedLength[128];

vec4* v  = (vec4*)vertices;
vec4* vp  = (vec4*)verticesPre;


vec4 wind = vec4(0.0f, 0.0f, 0.0f, 0.0f);
vec4 gravity = vec4(0.0f, 0.0f, 0.0f, 0.0f);

int numELCIter = 2;
int numLSCIter = 3;
int bCollision = true;    
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

vec4 Integrate(vec4 curPosition, vec4 oldPosition, vec4 force, int numVerticesInTheStrand, int vertexIndfloat,float dampingCoeff = 1.0f)
{  
    vec4 outputPos = curPosition;

    //force.xyz += gravity.xyz;
    outputPos.x = curPosition.x + (1.0f - dampingCoeff)*(curPosition.x - oldPosition.x) + force.x*timestep*timestep; 
    outputPos.y = curPosition.y + (1.0f - dampingCoeff)*(curPosition.y - oldPosition.y) + force.y*timestep*timestep; 
    outputPos.z = curPosition.z + (1.0f - dampingCoeff)*(curPosition.z - oldPosition.z) + force.z*timestep*timestep; 
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
        vec3 vecTangent = segN * dot(vec, segN);
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
void UpdateFinalVertexPositions(vec4 oldPosition, vec4 newPosition, int numVerticesInTheStrand, int vertexInd)
{ 
    if ( vertexInd < numVerticesInTheStrand )
    {
        vp[vertexInd] = oldPosition;
        v[vertexInd * 3] = newPosition;
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
    //int globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem;
    //CalcIndicesInVertexLevel(GIndex, GId.x, globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem);
    
    vec4 currentPos = vec4(0, 0, 0, 0); // position when this step starts. In other words, a position from the last step. 
    
    // Copy data into shared memory 
    //currentPos = sharedPos[vertexInd] = g_HairVertices[vertexInd * 3];
    currentPos = sharedPos[vertexInd] = v[vertexInd * 3];

    //GroupMemoryBarrierWithGroupSync();

    if ( vertexInd == 0 || vertexInd == 1 )
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
		sharedPos[vertexInd] = Integrate(currentPos, oldPos, force, NUM_VERTS_PER_STRAND, vertexInd, dampingCoeff); 
    
    // update global position buffers
	UpdateFinalVertexPositions(currentPos, sharedPos[vertexInd], NUM_VERTS_PER_STRAND, vertexInd);
}


//--------------------------------------------------------------------------------------
// 
//  LocalShapeConstraints
//
//  Compute shader to maintain the local shape constraints.
//
// One thread computes one strand.
//
//--------------------------------------------------------------------------------------
void LocalShapeConstraints(int vertexInd)
{
    //int local_id = GIndex; 
    //int group_id = GId.x;

    //int globalStrandIndex = THREAD_GROUP_SIZE*group_id + local_id;
    int numVerticesInTheStrand = NUM_VERTS_PER_STRAND;
    //int globalRootVertexIndex = numVerticesInTheStrand * globalStrandIndex;

    //--------------------------------------------
    // Local shape constraint for bending/twisting 
    //--------------------------------------------    
    for ( int iter = 0; iter < numLSCIter; iter++ )
    {
        vec4 pos_minus_one = v[vertexInd * 3];
        vec4 pos = v[(vertexInd + 1) * 3];
        vec4 pos_plus_one;
        int globalVertexIndex = 0;
        vec4 rotGlobal = GlobalFrames[vertexInd];

        //g_DebugBuffer[globalStrandIndex] = pos;
                
        for ( int localVertexIndex = 1; localVertexIndex < numVerticesInTheStrand-1; localVertexIndex++ )
        {
            //globalVertexIndex = globalRootVertexIndex + localVertexIndex;
            pos_plus_one = v[(localVertexIndex + 1) * 3];

            //--------------------------------
            // Update position i and i_plus_1
            //--------------------------------
            vec4 rotGlobalWorld = rotGlobal; 
            vec3 orgPos_i_plus_1_InLocalFrame_i = RefVector[localVertexIndex + 1]._xyz(); 
            vec3 orgPos_i_plus_1_InGlobalFrame = MultQuaternionAndVector(rotGlobalWorld, orgPos_i_plus_1_InLocalFrame_i) + pos._xyz();

            vec3 del = (orgPos_i_plus_1_InGlobalFrame - pos_plus_one._xyz()) * stiffnessLSC * 0.5f ;

            // Without the following line, sudden jerky movement can be observed when a new cell goes under simulation due to numeric error. 
            if ( del.Length() > 0.0015 )
            {
                if ( IsMovable(pos) )
                {
                    pos.x -= del.x;
                    pos.y -= del.y;
                    pos.z -= del.z;

                }
                
                pos.x -= del.x;
                pos.y -= del.y;
                pos.z -= del.z;
                

                if ( IsMovable(pos_plus_one) )
                {
                    pos_plus_one.x += del.x;
                    pos_plus_one.y += del.y;
                    pos_plus_one.z += del.z;
                }
            }
    
            //---------------------------
            // Update local/global frames
            //---------------------------
            vec4 invRotGlobalWorld = InverseQuaternion(rotGlobalWorld);   
            vec3 vec = normalize(pos_plus_one._xyz() - pos._xyz());     
    
            vec3 x_i_plus_1_frame_i = normalize(MultQuaternionAndVector(invRotGlobalWorld, vec));
            vec3 e = vec3(1.0f, 0, 0);
            vec3 rotAxis = cross(e, x_i_plus_1_frame_i);
    
            if ( rotAxis.Length() > 0.001 )
            {
                float angle_radian = acos(dot(e, x_i_plus_1_frame_i));
                rotAxis = normalize(rotAxis);

                vec4 localRot = MakeQuaternion(angle_radian, rotAxis);
                
                rotGlobal = MultQuaternionAndQuaternion(rotGlobal, localRot);
            }   

            v[(localVertexIndex) * 3].x = pos.x;
            v[(localVertexIndex) * 3].y = pos.y;    
            v[(localVertexIndex) * 3].z = pos.z;    

            v[(localVertexIndex + 1) * 3].x = pos_plus_one.x;
            v[(localVertexIndex + 1) * 3].y = pos_plus_one.y;
            v[(localVertexIndex + 1) * 3].z = pos_plus_one.z;

            pos_minus_one = pos;
            pos = pos_plus_one;
        }     
    }

    return;
}

//--------------------------------------------------------------------------------------
// 
//  LengthConstriantsAndWind
//
//  Compute shader to move the vertex position based on wind and maintains the lenght constraints.
//
// One thread computes one vertex.
//
//--------------------------------------------------------------------------------------
void LengthConstriantsAndWind(int vertexInd)
{
    //int globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem;
    //CalcIndicesInVertexLevel(GIndex, GId.x, globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem);
        
    int numOfStrandsPerThreadGroup = NUM_STRANDS_PER_GROUP;
        
    //------------------------------
    // Copy data into shared memory
    //------------------------------
    {
        sharedPos[vertexInd] = v[vertexInd * 3];
        sharedLength[vertexInd] = RestLength[vertexInd]; 
        //g_DebugBuffer[globalVertexIndex] = g_HairRestLengthSRV[globalVertexIndex + bladeOffset];
    }
    
    
    //GroupMemoryBarrierWithGroupSync();

    ////-------
    //// Wind
    ////-------
    //if ( IsMovable(sharedPos[indexForSharedMem])  )
    //{
    //  vec4 ran = g_Random[globalVertexIndex];
    //  float dx0 = ran.y*sin( TimeVals.y * ran.x ) * 0.004;
    //  float dx1 = sin( TimeVals.y + ran.z) * 0.0007;
    //  sharedPos[indexForSharedMem]._xyz() += vec4((dx0+dx1), 0, 0,  0);
    //}
    
    //GroupMemoryBarrierWithGroupSync();

    //----------------------------
    // Enforce length constraints
    //----------------------------
    int a = floor(NUM_VERTS_PER_STRAND/2.0f);
    int b = floor((NUM_VERTS_PER_STRAND-1)/2.0f); 
        
    for ( int iterationE=0; iterationE < numELCIter; iterationE++ )
    {       
        int sharedIndex = 2*vertexInd * numOfStrandsPerThreadGroup + vertexInd;

        if( vertexInd < a )
            ApplyDistanceConstraint(sharedPos[sharedIndex], sharedPos[sharedIndex+numOfStrandsPerThreadGroup], sharedLength[sharedIndex].x);

        //GroupMemoryBarrierWithGroupSync();

        if( vertexInd < b )
            ApplyDistanceConstraint(sharedPos[sharedIndex+numOfStrandsPerThreadGroup], sharedPos[sharedIndex+numOfStrandsPerThreadGroup*2], sharedLength[sharedIndex+numOfStrandsPerThreadGroup].x);

        //GroupMemoryBarrierWithGroupSync();
    }

    //---------------------------------------
    // update global position buffers
    //---------------------------------------
    if ( vertexInd < NUM_VERTS_PER_STRAND )
    v[vertexInd * 3] = sharedPos[vertexInd];
    
    return;
}


void CollisionAndTangents(int vertexInd)
{
    //int globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem;
    //CalcIndicesInVertexLevel(GIndex, GId.x, globalStrandIndex, localStrandIndex, globalVertexIndex, localVertexIndex, numVerticesInTheStrand, indexForSharedMem);

    int numOfStrandsPerThreadGroup = NUM_STRANDS_PER_GROUP;

    //------------------------------
    // Copy data into shared memory
    //------------------------------
	if (vertexInd < NUM_VERTS_PER_STRAND )
    {
        sharedPos[vertexInd] = v[vertexInd * 3];
    }

    vec4 oldPos = vp[vertexInd];
        
    //GroupMemoryBarrierWithGroupSync();
    
    // collision handling with spheres (rocks)
    if ( bCollision )
    {
        for ( int i = 0; i < NUM_OF_ROCKS; i++ )
        {
			sp = sp + vec4(0.0f, 0.0f, 0.05f, 0.0f);
            sharedPos[vertexInd].x = SphereCollision(sharedPos[vertexInd], sp).x;
            sharedPos[vertexInd].y = SphereCollision(sharedPos[vertexInd], sp).y;
            sharedPos[vertexInd].z = SphereCollision(sharedPos[vertexInd], sp).z;
        }
    }

    //GroupMemoryBarrierWithGroupSync();
    
    //-----------------------------
    // Compute normal and tangent
    //-----------------------------
    vec3 tangent;
    vec3 binormal = Binormal[0]._xyz();
    vec3 normal;

    if ( vertexInd == 0 )
    {
        tangent = sharedPos[vertexInd+numOfStrandsPerThreadGroup]._xyz() - sharedPos[vertexInd]._xyz();
    }
	else if ( vertexInd == NUM_VERTS_PER_STRAND - 1 )
    {
        tangent = sharedPos[vertexInd]._xyz() - sharedPos[vertexInd-numOfStrandsPerThreadGroup]._xyz();
    }
    else 
    {
        vec3 t0 = sharedPos[vertexInd+numOfStrandsPerThreadGroup]._xyz() - sharedPos[vertexInd]._xyz();
        vec3 t1 = sharedPos[vertexInd]._xyz() - sharedPos[vertexInd-numOfStrandsPerThreadGroup]._xyz();
        tangent = t0 + t1;
    }

    normal =  cross(tangent, binormal);
    float len = std::max(normal.Length(), 1e-7f);
    normal = normal / len;

    v[vertexInd * 3 + 2].x = normalize(tangent).x;
    v[vertexInd * 3 + 2].y = normalize(tangent).y;
    v[vertexInd * 3 + 2].z = normalize(tangent).z;
    v[vertexInd * 3 + 2].w = 1.0f - ballLeftTime;

    v[vertexInd * 3 + 1].x = normal.x;
    v[vertexInd * 3 + 1].y = normal.y;
    v[vertexInd * 3 + 1].z = normal.z;

    //---------------------------------------
    // update global position buffers
    //---------------------------------------
	if ( vertexInd < NUM_VERTS_PER_STRAND )
        {
            // This code would blend position towards initial based on fractional time 
            // left after leaving the cell.  However, it would then feed into the simulation again,
            // turning this into something resembling a ramped constraint. This is somewhat redundant with 
            // the ramped local constraint idea that Dongsoo suggested.
            // g_HairVertices[globalVertexIndex * 3] = sharedPos[indexForSharedMem] * (1.0f - ballLeftTime) + g_InitialHairPositions[globalVertexIndex] * ballLeftTime;
                
            v[vertexInd * 3] = sharedPos[vertexInd];
        }
    
    return;
}
