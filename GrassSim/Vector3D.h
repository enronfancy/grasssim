//--------------------------------------------------------------------------------------
// File: Vector3D.h
//--------------------------------------------------------------------------------------

#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

#include <cmath>
#include "Matrix44.h"

class CVector3D
{
public:
	float m_X;
	float m_Y;
	float m_Z;

	CVector3D() 
	{
		m_X = 0.0;
		m_Y = 0.0;
		m_Z = 0.0;
	}

	CVector3D(float x, float y, float z) { m_X = x; m_Y = y; m_Z = z; };
	CVector3D(float val) { m_X = val; m_Y = val; m_Z = val; };
	CVector3D(const CVector3D& other);	
	CVector3D(const CVector3D& begin, const CVector3D& end);
	virtual ~CVector3D() {};

public:
	CVector3D& Normalize();
	CVector3D NormalizeOther() const;
	CVector3D Cross(const CVector3D& v) const { return CVector3D(m_Y*v.m_Z - m_Z*v.m_Y, m_Z*v.m_X - m_X*v.m_Z, m_X*v.m_Y - m_Y*v.m_X); };
	float Dot(const CVector3D& v) const { return m_X * v.m_X + m_Y * v.m_Y + m_Z * v.m_Z; };	
	float Length() const { return sqrt(m_X*m_X + m_Y*m_Y + m_Z*m_Z); };
	float LengthSqr() const { return (m_X*m_X + m_Y*m_Y + m_Z*m_Z); };
	void Set(float x, float y, float z) { m_X = x; m_Y = y; m_Z = z; };
	
	void RotateW(const CVector3D& axis, float ang);
	void RotateW(CMatrix44 mat44);
	
	const float& operator[](unsigned int i) const
	{
		if ( i == 0 )
			return m_X;
		else if ( i == 1)
			return m_Y;
		else /*if ( i == 2 )*/
			return m_Z;
	}

	float& operator[](unsigned int i) 
	{
		if ( i == 0 )
			return m_X;
		else if ( i == 1)
			return m_Y;
		else /*if ( i == 2 )*/
			return m_Z;
	}

	CVector3D& operator=(const CVector3D& other);
	CVector3D& operator=(float val);
	bool operator!=(float val) const;
	bool operator<(float val) const;
	bool operator>(float val) const;
	bool operator==(float val) const;
	bool operator==(const CVector3D& other) const;
	bool operator!=(const CVector3D& other) const;
	CVector3D& operator-=(const CVector3D& other);
	CVector3D& operator+=(const CVector3D& other);
	CVector3D& operator*=(float val);
	CVector3D operator-(const CVector3D& other) const;
	CVector3D operator+(const CVector3D& other) const;
	CVector3D operator/(float val) const;
	CVector3D operator*(float val) const;
	float operator*(const CVector3D& other) const;
	CVector3D operator/(const CVector3D& other) const;

	friend CVector3D operator*(float val, const CVector3D& other);
	friend CVector3D operator-(const CVector3D& other);

};

#endif // __VECTOR3D_H__