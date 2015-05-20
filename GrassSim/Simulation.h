#ifndef _SIM_H
#define _SIM_H

#include "Grass.h"
//#include "myFloat.h"
#include "mathutils.h"
#include <math.h>

bool IsMovable(vec4 particle);

vec2 ConstraintMultiplier(vec4 particle0, vec4 particle1);

vec4 MakeQuaternion(float angle_radian, vec3 axis);

vec4 InverseQuaternion(vec4 q);

vec3 MultQuaternionAndVector(vec4 q, vec3 v);

vec4 MultQuaternionAndQuaternion(vec4 qA, vec4 qB);

vec4 NormalizeQuaternion(vec4& q);

void ApplyDistanceConstraint(vec4& pos0, vec4& pos1, float targetDistance, float stiffness = 1.0);

void CalcIndicesInVertexLevel(int local_id, int group_id, int &globalStrandIndex, int &localStrandIndex, int &globalVertexIndex, int &localVertexIndex, int &numVerticesInTheStrand, int &indexForSharedMem);

vec4 Integrate(vec4 curPosition, vec4 oldPosition, vec4 force, int numVerticesInTheStrand, int vertexIndfloat,float dampingCoeff = 1.0f);

struct CollisionCapsule;

vec3 CapsuleCollision(vec4 curPosition, vec4 oldPosition, CollisionCapsule cc, float friction = 0.4f);

vec3 SphereCollision(vec4 curPosition, vec4 sphere);

void UpdateFinalVertexPositions(vec4 oldPosition, vec4 newPosition, int numVerticesInTheStrand, int vertexInd);

//Simulation functions
void IntegrationAndGlobalShapeConstraints(int vertexInd);

void LocalShapeConstraints(int vertexInd);

void LengthConstriantsAndWind(int vertexInd);

void CollisionAndTangents(int vertexInd);


#endif