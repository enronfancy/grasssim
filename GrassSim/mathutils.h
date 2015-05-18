#ifndef _MATHUTILS_H
#define _MATHUTILS_H
#include <math.h>
#include <iostream>
#include <random>

#define kPI 3.14159265358979323846f
#define kTwoPI (kPI * 2.f)
#define kOneOverPI (1.f/kPI)
#define kOneOverTwoPI (1.f/(2.f*kPI))
#define NEXTMULTIPLEOF(num, alignment) (((num)/(alignment) + (((num)%(alignment)==0)?0:1))*(alignment))
const static float kSqrt2 = sqrt(2.f);

#define tovec3(x)         vec3( x[0], x[1], x[2] )
#define vec3toarray(x,v)  { x[0] = v.x; x[1] = v.y; x[2] = v.z; }
#define vec3c(x, y, z)    vec3( x/255.f, y/255.f, z/255.f )

namespace kiMath
{
   inline bool closeToZero3(float x)
   {
      return fabs(x) < 0.0001f;
   }

   inline float kRadianToDegrees(float v)
   {
      return (v)*(180.f/kPI);
   }

   inline float kDegreesToRadian(float v)
   {
      return (v)*(kPI/180.f);
   }

   struct Vec3
   {
      float x;
      float y;
      float z;
   };

   inline float Min(float x, float y)
   {
      return (x <= y) ? x : y;
   }

   inline float Max(float x, float y)
   {
      return (x >= y) ? x : y;
   }

   inline int Min(int x, int y)
   {
      return (x <= y) ? x : y;
   }

   inline int Max(int x, int y)
   {
      return (x >= y) ? x : y;
   }

   inline float Clamp(float x, float from = 0.f, float to = 1.f)
   {
      if(x > to)      x = to;
      else if(x < from) x = from;

      return x;
   }

   inline float Saturate(float x)
   {
      if(x > 1.f)      x = 1.f;
      else if(x < 0.f) x = 0.f;

      return x;
   }

   inline float Lerp(float from, float to, float t)
   {
      return from + (to-from)*t;
   }

   inline float SmoothStep(float from, float to, float x)
   {
      float clamped = Clamp(x,from,to);
      float frac    = (clamped-from)/(to-from);
      return frac;
   }

   inline float ContrainToCircleRadians(float x)
   {
      if(x > kTwoPI) x = (-kTwoPI + (x-kTwoPI));
      else if(x < -kTwoPI) x = (kTwoPI + (x + kTwoPI));
      return x;
   }

   inline void SRand( int seed )
   {
      srand( seed );
   }

   inline float RandomFloat( float start, float end)
   {
      float randomVal = (float)rand()/RAND_MAX;
      return start + (end-start)*randomVal;
   }

   inline int RandomInt( int start, int end)
   {
      int range = end-start;
      return start + rand()%(range + 1);
   }

   inline float Wrap(float x, float dx, float wrapVal = 1.f, float start = 0.f)
   {
      x += dx;
      if(x <= wrapVal) return x;
      return start + (x - wrapVal);
   }

 
   class vec2 
   {
   public:

      float x;
      float y;

      vec2() { };

      vec2(float _x, float _y)
      {
         x = _x;
         y = _y;
      }

      vec2(const vec2 &a)
      {
         x = a.x;
         y = a.y;
      }

      void Set(float _x, float _y)
      {
         x = _x;
         y = _y;
      }

      vec2 operator*(const float a);

	  float operator[](const int index) const 
	  {
		return (&x)[index];
	  }
   };

   inline vec2 vec2::operator*(const float a) 
   {
      return vec2( x*a, y*a );
   }

   class vec3 
   {
   public:

      float x;
      float y;
      float z;

	  vec3();
      vec3(const vec3 &a);

      vec3(float _x, float _y, float _z);
      void Set(float _x, float _y, float _z);

      vec3 operator+(const vec3 &a) const;
      vec3 operator-( const vec3 &a );

      float operator[] (int index) const;
      vec3 operator-() const;
      vec3 operator*=(const float a);
      vec3 operator*(const float a);
      vec3 operator*(const vec3& a);

      vec3 operator/(const float a);

      vec3 operator+=(const vec3 &a);
      vec3 operator-=(const vec3 &a);
      vec3 operator=(const vec3 &a);
      vec3 operator=(const float &a);
      vec3 operator=(const float* a);

      float  Dot(const vec3 &a) const; 
      vec3     Cross(const vec3 &a) const;  
      float  Normalize(void);
      float  Length(void) const;
   };

   inline vec3::vec3()
   {
	   x = y = z = 0;
   }


   inline vec3::vec3(const vec3 &a)
   {
      x = a.x;
      y = a.y;
      z = a.z;
   }

   inline vec3::vec3(float _x, float _y, float _z)
   {
      x = _x;
      y = _y;
      z = _z;
   }

   inline void vec3::Set(float _x, float _y, float _z)
   {
      x = _x;
      y = _y;
      z = _z;
   }

   inline vec3 vec3::operator+(const vec3 &a) const 
   {
      return vec3(x + a.x, y + a.y, z + a.z);
   }

   inline vec3 vec3::operator-(const vec3 &a)
   {
      return vec3(x - a.x, y - a.y, z - a.z);
   }

   inline float vec3::operator[](const int index) const 
   {
      return (&x)[index];
   }

   inline vec3 vec3::operator-() const 
   {
      return vec3( -x, -y, -z );
   }

   inline vec3 vec3::operator*=(const float a) 
   {
      x *= a;
      y *= a;
      z *= a;
      return *this;
   }

   inline vec3 vec3::operator*(const float a) 
   {
      return vec3( x*a, y*a, z*a );
   }

   inline vec3 vec3::operator/(const float a)
   {
      if( abs(a) < 1e-5 ) return vec3(0,0,0);
      return vec3( x/a, y/a, z/a );
   }

   inline vec3 vec3::operator*(const vec3& a)  
   {
      return vec3( x*a.x, y*a.y, z*a.z );
   }

   inline vec3 vec3::operator+=(const vec3 &a)
   {
      x += a.x;
      y += a.y;
      z += a.z;

      return *this;
   }

   inline vec3 vec3::operator=(const vec3 &a)
   {
      x = a.x;
      y = a.y;
      z = a.z;

      return *this;
   }

   inline vec3 vec3::operator=(const float* a)
   {
      x = a[0];
      y = a[1];
      z = a[2];

      return *this;
   }

   inline vec3 vec3::operator=(const float &a)
   {
      x = a;
      y = a;
      z = a;

      return *this;
   }

   inline vec3 vec3::operator-=(const vec3 &a) 
   {
      x -= a.x;
      y -= a.y;
      z -= a.z;
      return *this;
   }

   inline float vec3::Dot(const vec3 &a) const 
   {
      return x*a.x + y*a.y + z*a.z;
   }

   inline vec3 vec3::Cross(const vec3 &a) const 
   {
      return vec3( y*a.z - z*a.y, z*a.x - x*a.z, x*a.y - y*a.x );
   }

   inline float vec3::Length(void) const 
   {
      return (float)sqrt( x*x + y*y + z*z );
   }

   inline float vec3::Normalize(void) 
   {
      float len = this->Length();
      if( len != 0.f ) 
      {
         float invLen = 1.0f / len;
         x *= invLen;
         y *= invLen;
         z *= invLen;
      }
      return len;
   } 

   inline vec3 Saturate(vec3 v)
   {
      vec3 res;

      res.x = Saturate(v.x);
      res.y = Saturate(v.y);
      res.z = Saturate(v.z);

      return v;
   }

   inline vec3 LinearToSRGB(vec3& v) 
   {
      vec3 r;

      r.x = Saturate( pow(v.x, 0.4545454545f) );
      r.y = Saturate( pow(v.y, 0.4545454545f) );
      r.z = Saturate( pow(v.z, 0.4545454545f) );

      return r;
   }

   inline vec3 SRGBToLinear(vec3& v) 
   {
      vec3 r;

      r.x = Saturate( pow(v.x, 2.2f) );
      r.y = Saturate( pow(v.y, 2.2f) );
      r.z = Saturate( pow(v.z, 2.2f) );

      return r;
   }

   class vec4 
   {
   public:

      float x;
      float y;
      float z;
      float w;

      vec4() { };

      vec4(const vec3 &a);
      vec4(const vec4 &a);
      vec4(float _x, float _y, float _z, float _w);
      void Set(float _x, float _y, float _z, float _w);

      vec4 operator+(const vec4 &b);
      vec4 operator-( const vec4 &a );

      float operator[] (int index) const;
      vec4 operator-();
      vec4 operator*=(const float a);
      vec4 operator*(const float a);

      vec4 operator+=(const vec4 &a);
      vec4 operator-=(const vec4 &a);
      vec4 operator=(const vec4 &a);
      vec4 operator=(const float &a);
      vec4 operator=(const float* a);

      vec3 _xyz();
   };

   inline vec4::vec4(const vec3 &a)
   {
      x = a.x;
      y = a.y;
      z = a.z;
      w = 1.f;
   }

   inline vec4::vec4(const vec4 &a)
   {
      x = a.x;
      y = a.y;
      z = a.z;
      w = a.w;
   }

   inline vec4::vec4(float _x, float _y, float _z, float _w)
   {
      x = _x;
      y = _y;
      z = _z;
      w = _w;
   }

   inline vec3 vec4::_xyz()
   {
      vec3 v = vec3(x,y,z);
      return v;
   }

   inline void vec4::Set(float _x, float _y, float _z, float _w)
   {
      x = _x;
      y = _y;
      z = _z;
      w = _w;
   }

   inline float vec4::operator[](const int index) const 
   {
      return (&x)[index];
   }

   inline vec4 vec4::operator*=(const float a) 
   {
      x *= a;
      y *= a;
      z *= a;
      return *this;
   }

   inline vec4 vec4::operator*(const float a) 
   {
      return vec4( x*a, y*a, z*a, w );
   }

   inline vec4 vec4::operator+=(const vec4 &a)
   {
      x += a.x;
      y += a.y;
      z += a.z;
      w = a.w;

      return *this;
   }

   inline vec4 vec4::operator=(const vec4 &a)
   {
      x = a.x;
      y = a.y;
      z = a.z;
      w = a.w;
      return *this;
   }

   inline vec4 vec4::operator=(const float* a)
   {
      x = a[0];
      y = a[1];
      z = a[2];
      w = a[3];
      return *this;
   }

   inline vec4 vec4::operator=(const float &a)
   {
      x = a;
      y = a;
      z = a;
      w = a;

      return *this;
   }

   inline vec4 vec4::operator-=(const vec4 &a) 
   {
      x -= a.x;
      y -= a.y;
      z -= a.z;
      return *this;
   }

   inline vec4 vec4::operator-(const vec4 &b) 
   {
      vec4 val;

      val.x = x - b.x;
      val.y = y - b.y;
      val.z = z - b.z;
      val.w = w - b.w;

      return val;
   }

   inline vec4 vec4::operator+(const vec4 &b) 
   {
      vec4 val;

      val.x = x + b.x;
      val.y = y + b.y;
      val.z = z + b.z;
      val.w = w + b.w;

      return val;
   }

   inline vec4 vec4::operator-() 
   {
      return vec4( -x, -y, -z, -w );
   }


#define VectorCopy(a,b) a[0] = b[0]; a[1] = b[1]; a[2] = b[2];
#define VectorZero(a)   a[0] = 0.f; a[1] = 0.f; a[2] = 0.f;

   inline void Cross(float* result, float* a, float* b)
   {
      float r[3];
      r[0] = a[1]*b[2] - a[2]*b[1];
      r[1] = a[2]*b[0] - a[0]*b[2];
      r[2] = a[0]*b[1] - a[1]*b[0];
      VectorCopy(result,r);
   }

   inline float Dot(float* a, float* b)
   {
      return( a[0]*b[0] + a[1]*b[1] + a[2]*b[2] );
   }

   inline float VecMul(float* vec, float a)
   {
      vec[0] *= a;
      vec[1] *= a;
      vec[2] *= a;
   }

   inline float VecAdd(float* result, float* a, float* b)
   {
      result[0] = a[0] + b[0];
      result[1] = a[1] + b[1];
      result[2] = a[2] + b[2];
   }

   inline vec3 Lerp(vec3& from, vec3& to, float t)
   {
      return from + (to-from)*t;
   }

   class PolarCoord
   {
   public:
      float polar;
      float azim;
      float r;

      PolarCoord()
      {
         polar = 0;
         azim  = 0;
         r     = 1;
      }

      PolarCoord(float _r, float _polar, float _azim)
      {
         r     = _r;
         polar = _polar;
         azim  = _azim;
      }

      vec3 AsVec()
      {
         vec3 v;

         v.x = r * -sin(azim) * cos(polar);
         v.y = r * sin(polar);
         v.z = r * -cos(azim) * cos(polar);

         return v;
      }

      void SetLength(float _r)
      {
         r = _r;
      }

      void SetAngles(float _polar, float _azim)
      {
         polar = _polar;
         azim  = _azim;
      }

      void SetPolar(float _polar)
      {
         polar = _polar;
      }

      void SetAzim(float _azim)
      {
         azim  = _azim;
      }
   };

   inline PolarCoord VecToPolar(vec3 v)
   {
	   PolarCoord p;

	   p.r     = v.Length();
	   p.azim  = atan2f( -v.x,  -v.z );
	   p.polar = (kPI/2.f) - acos( v.y / p.r );

	   return p;
   }

   inline int Factorial(int x)
   {
      int res;
      if( x == 0 || x == -1) return(1);
      res = x;
      while( ( x -= 1 ) > 0) res *= x;
      return res;
   }

   inline int DoubleFac(int x)
   {
      int res;
      if( x == 0 || x == -1) return(1);
      res = x;
      while( ( x -= 2 ) > 0) res *= x;
      return res;
   }

   vec3 Centroid( int numPoints, vec3* pnts );


#define MatrixSetC1(a,v) {	a.m[1] =  v[0]; \
   a.m[5] =  v[1]; \
   a.m[9] =  v[2]; }

#define MatrixSetC2(a,v) {	a.m[2]  =  v[0]; \
   a.m[6]  =  v[1]; \
   a.m[10] =  v[2]; }

#define MatrixSetTrans(a,v) {	a.m[3]  =  v[0]; \
   a.m[7]  =  v[1]; \
   a.m[11] =  v[2]; }

#define MatrixSetC0Vec(a,v) {	a.m[0] =  v.x; \
   a.m[4] =  v.y; \
   a.m[8] =  v.z; 

#define MatrixSetC1Vec(a,v) {	a.m[1] =  v.x; \
   a.m[5] =  v.y; \
   a.m[9] =  v.z; }

#define MatrixSetC2Vec(a,v) {		a.m[2]  =  v.x; \
   a.m[6]  =  v.y; \
   a.m[10] =  v.z; }

#define MatrixSetTransVec(a,v) {	a.m[3]  =  v.x; \
   a.m[7]  =  v.y; \
   a.m[11] =  v.z; }

   struct Matrix4x4
   {
      float m[16];

      Matrix4x4() {}

      Matrix4x4( bool setIdentity ) 
      {
         if( setIdentity ) Identity();
      }

      void Identity()
      {
         m[0] =  1.f;  m[1]  = 0.f;   m[2]  = 0.f;    m[3] = 0.f;    
         m[4] =  0.f;  m[5]  = 1.f;   m[6]  = 0.f;    m[7] = 0.f;    
         m[8] =  0.f;  m[9]  = 0.f;   m[10] = 1.f;    m[11] = 0.f;   
         m[12] = 0.f;  m[13] = 0.f;   m[14] = 0.f;    m[15] = 1.f;  
      }

      void SetRight(vec3 v)
      {
         m[0] =  v.x;
         m[4] =  v.y; 
         m[8] =  v.z; 
      }


      void SetUp(vec3 v)
      {
         m[1] =  v.x;
         m[5] =  v.y; 
         m[9] =  v.z; 
      }


      void SetForward(vec3 v)
      {
         m[2]  =  v.x;
         m[6]  =  v.y; 
         m[10] =  v.z; 
      }

      void SetTrans(vec3 v)
      {
         m[3]  =  v.x;
         m[7]  =  v.y; 
         m[11] =  v.z; 
      }

      vec3 GetTrans()
      {
         return( vec3( m[3], m[7], m[11]) );
      }

      vec3 GetRight()
      {
         return( vec3( m[0], m[4], m[8]) );
      }

      vec3 GetUp()
      {
         return( vec3( m[1], m[5], m[9]) );
      }

      vec3 GetForward()
      {
         return( vec3( m[2], m[6], m[10]) );
      }

	  vec3 GetEulers()
	  {
		  vec3 f       = -GetForward();
		  PolarCoord p = VecToPolar( f );
		  return vec3( kRadianToDegrees( p.polar ),  kRadianToDegrees( p.azim ), 0 );
	  }

      void Transpose()
      {
         Matrix4x4 mat;

         for(int i=0; i<4; i++)
         {
            for(int j=0; j<4; j++)
            {
               mat.m[i*4 + j] = m[j*4 + i];
            }
         }
         memcpy(m, mat.m, sizeof(float)*16);
      }

      Matrix4x4 Matrix4x4::operator=(const Matrix4x4 &a)
      {
         memcpy(this->m,a.m,sizeof(float)*16);
         return *this;
      }

      Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &b)
      {
         Matrix4x4 result;

         result.m[0] = m[0]*b.m[0] + m[1]*b.m[4] + m[2]*b.m[8] + m[3]*b.m[12];
         result.m[1] = m[0]*b.m[1] + m[1]*b.m[5] + m[2]*b.m[9] + m[3]*b.m[13];
         result.m[2] = m[0]*b.m[2] + m[1]*b.m[6] + m[2]*b.m[10] + m[3]*b.m[14];
         result.m[3] = m[0]*b.m[3] + m[1]*b.m[7] + m[2]*b.m[11] + m[3]*b.m[15];

         result.m[4] = m[4]*b.m[0] + m[5]*b.m[4] + m[6]*b.m[8] +  m[7]*b.m[12];
         result.m[5] = m[4]*b.m[1] + m[5]*b.m[5] + m[6]*b.m[9] +  m[7]*b.m[13];
         result.m[6] = m[4]*b.m[2] + m[5]*b.m[6] + m[6]*b.m[10] + m[7]*b.m[14];
         result.m[7] = m[4]*b.m[3] + m[5]*b.m[7] + m[6]*b.m[11] + m[7]*b.m[15];

         result.m[8]  = m[8]*b.m[0] + m[9]*b.m[4] + m[10]*b.m[8]  + m[11]*b.m[12];
         result.m[9]  = m[8]*b.m[1] + m[9]*b.m[5] + m[10]*b.m[9]  + m[11]*b.m[13];
         result.m[10] = m[8]*b.m[2] + m[9]*b.m[6] + m[10]*b.m[10] + m[11]*b.m[14];
         result.m[11] = m[8]*b.m[3] + m[9]*b.m[7] + m[10]*b.m[11] + m[11]*b.m[15];

         result.m[12] = m[12]*b.m[0] + m[13]*b.m[4] + m[14]*b.m[8] +  m[15]*b.m[12];
         result.m[13] = m[12]*b.m[1] + m[13]*b.m[5] + m[14]*b.m[9] +  m[15]*b.m[13];
         result.m[14] = m[12]*b.m[2] + m[13]*b.m[6] + m[14]*b.m[10] + m[15]*b.m[14];
         result.m[15] = m[12]*b.m[3] + m[13]*b.m[7] + m[14]*b.m[11] + m[15]*b.m[15];

		 return result;
      }

      void SetRight(float* v)  { SetRight(vec3(v[0],v[1],v[2])); }
      void SetUp(float* v)  { SetUp(vec3(v[0],v[1],v[2])); }
      void SetForward(float* v)  { SetForward(vec3(v[0],v[1],v[2])); }
      void SetTrans(float* v) { SetTrans(vec3(v[0],v[1],v[2])); }
   };

   struct Matrix3x3
   {
      float m[9];

      void Identity()
      {
         m[0] =  1.f;  m[1]  = 0.f;   m[2]  = 0.f;   
         m[4] =  0.f;  m[5]  = 1.f;   m[6]  = 0.f;   
         m[8] =  0.f;  m[9]  = 0.f;   m[10] = 1.f;   
      }

      void SetRight(vec3 v)
      {
         m[0] =  v.x;
         m[3] =  v.y; 
         m[6] =  v.z; 
      }


      void SetUp(vec3 v)
      {
         m[1] =  v.x;
         m[4] =  v.y; 
         m[7] =  v.z; 
      }

      void SetForward(vec3 v)
      {
         m[2]  =  v.x;
         m[5]  =  v.y; 
         m[8] =  v.z; 
      }

      void Transpose()
      {
         Matrix3x3 mat;

         for(int i=0; i<3; i++)
         {
            for(int j=0; j<3; j++)
            {
               mat.m[i*3 + j] = m[j*3 + i];
            }
         }
         memcpy(m, mat.m, sizeof(float)*9);
      }

      void SetRight(float* v)  { SetRight(vec3(v[0],v[1],v[2])); }
      void SetUp(float* v)  { SetUp(vec3(v[0],v[1],v[2])); }
      void SetForward(float* v)  { SetForward(vec3(v[0],v[1],v[2])); }
   };

   struct Matrix2x2
   {
      float m[4];

      void Identity()
      {
         m[0] =  1.f;  m[1] = 0.f;  
         m[2] =  0.f;  m[3] = 1.f;  
      }

      void SetRight(vec3 v)
      {
         m[0] =  v.x;
         m[2] =  v.y; 
      }

      void SetUp(vec3 v)
      {
         m[1] =  v.x;
         m[3] =  v.y; 
      }

      void SetRight(float* v)  { SetRight(vec3(v[0],v[1],v[2])); }
      void SetUp(float* v)  { SetUp(vec3(v[0],v[1],v[2])); }
   };

   inline void Matrix2x2MultiplyVec2(vec3& outP,Matrix2x2& mat,vec3& p)
   {
      outP.x = mat.m[0]*p.x + mat.m[1]*p.y;
      outP.y = mat.m[2]*p.x + mat.m[3]*p.y;
   }

   inline void Matrix3x3MultiplyVec3(vec3& outP,Matrix3x3& mat,vec3& p)
   {
      outP.x = mat.m[0]*p.x + mat.m[1]*p.y + mat.m[2]*p.z;
      outP.y = mat.m[3]*p.x + mat.m[4]*p.y + mat.m[5]*p.z;
      outP.z = mat.m[6]*p.x + mat.m[7]*p.y + mat.m[8]*p.z;
   }

   inline void Matrix4x4MultVec3(vec3& outP,Matrix4x4& mat,vec3& p)
   {
	   vec4 p1(p.x,p.y,p.z,1.f);

	   vec4 tmp;

	   tmp.x = mat.m[0]*p1.x + mat.m[1]*p1.y   + mat.m[2]*p1.z  + mat.m[3]*p1.w;
	   tmp.y = mat.m[4]*p1.x + mat.m[5]*p1.y   + mat.m[6]*p1.z  + mat.m[7]*p1.w;
	   tmp.z = mat.m[8]*p1.x + mat.m[9]*p1.y   + mat.m[10]*p1.z + mat.m[11]*p1.w;
	   tmp.w = mat.m[12]*p1.x + mat.m[13]*p1.y + mat.m[14]*p1.z + mat.m[15]*p1.w;

	   outP = vec3(tmp.x,tmp.y,tmp.z);
   }

   inline void Matrix4x4MultVec4(vec4& outP,Matrix4x4& mat,vec4& p1)
   {
	   vec4 tmp;

	   tmp.x = mat.m[0]*p1.x + mat.m[1]*p1.y   + mat.m[2]*p1.z  + mat.m[3]*p1.w;
	   tmp.y = mat.m[4]*p1.x + mat.m[5]*p1.y   + mat.m[6]*p1.z  + mat.m[7]*p1.w;
	   tmp.z = mat.m[8]*p1.x + mat.m[9]*p1.y   + mat.m[10]*p1.z + mat.m[11]*p1.w;
	   tmp.w = mat.m[12]*p1.x + mat.m[13]*p1.y + mat.m[14]*p1.z + mat.m[15]*p1.w;

	   outP = tmp;
   }

   inline void Matrix4x4Transpose(Matrix4x4& to, Matrix4x4& from)
   {
      Matrix4x4 mat;
      for(int i=0; i<4; i++)
      {
         for(int j=0; j<4; j++)
         {
            mat.m[i*4 + j] = from.m[j*4 + i];
         }
      }
      memcpy(to.m, mat.m, sizeof(float)*16);
   }

   inline void MatrixCopy4x4(Matrix4x4& to, Matrix4x4& from)
   {
      memcpy(to.m,from.m,sizeof(float)*16);
   }


   inline void MatrixSetRotationX(Matrix4x4& result, float angle)
   {
      float cosA = cos(kDegreesToRadian(angle));
      float sinA = sin(kDegreesToRadian(angle));

      result.Identity();
      result.SetRight(   vec3(1,0,0) );
      result.SetUp(      vec3(0, cosA, sinA) );
      result.SetForward( vec3(0, -sinA, cosA) );
   }

   inline void MatrixSetRotationY(Matrix4x4& result, float angle)
   {
      float cosA = cos(kDegreesToRadian(angle));
      float sinA = sin(kDegreesToRadian(angle));

      result.Identity();
      result.SetRight(   vec3(cosA, 0, -sinA) );
      result.SetUp(      vec3(0,     1, 0) );
      result.SetForward( vec3(sinA, 0, cosA) );
   }

   inline void MatrixSetRotationZ(Matrix4x4& result, float angle)
   {
      float cosA = cos(kDegreesToRadian(angle));
      float sinA = sin(kDegreesToRadian(angle));

      result.Identity();
      result.SetRight(   vec3(cosA, sinA,0) );
      result.SetUp(      vec3(-sinA,cosA,0) );
      result.SetForward( vec3(0,    0,   1) );
   }
   

   inline void MatrixMultiply4x4(Matrix4x4& result, Matrix4x4& a, Matrix4x4& b)
   {
      result.m[0] = a.m[0]*b.m[0] + a.m[1]*b.m[4] + a.m[2]*b.m[8] + a.m[3]*b.m[12];
      result.m[1] = a.m[0]*b.m[1] + a.m[1]*b.m[5] + a.m[2]*b.m[9] + a.m[3]*b.m[13];
      result.m[2] = a.m[0]*b.m[2] + a.m[1]*b.m[6] + a.m[2]*b.m[10] + a.m[3]*b.m[14];
      result.m[3] = a.m[0]*b.m[3] + a.m[1]*b.m[7] + a.m[2]*b.m[11] + a.m[3]*b.m[15];

      result.m[4] = a.m[4]*b.m[0] + a.m[5]*b.m[4] + a.m[6]*b.m[8] +  a.m[7]*b.m[12];
      result.m[5] = a.m[4]*b.m[1] + a.m[5]*b.m[5] + a.m[6]*b.m[9] +  a.m[7]*b.m[13];
      result.m[6] = a.m[4]*b.m[2] + a.m[5]*b.m[6] + a.m[6]*b.m[10] + a.m[7]*b.m[14];
      result.m[7] = a.m[4]*b.m[3] + a.m[5]*b.m[7] + a.m[6]*b.m[11] + a.m[7]*b.m[15];

      result.m[8]  = a.m[8]*b.m[0] + a.m[9]*b.m[4] + a.m[10]*b.m[8]  + a.m[11]*b.m[12];
      result.m[9]  = a.m[8]*b.m[1] + a.m[9]*b.m[5] + a.m[10]*b.m[9]  + a.m[11]*b.m[13];
      result.m[10] = a.m[8]*b.m[2] + a.m[9]*b.m[6] + a.m[10]*b.m[10] + a.m[11]*b.m[14];
      result.m[11] = a.m[8]*b.m[3] + a.m[9]*b.m[7] + a.m[10]*b.m[11] + a.m[11]*b.m[15];

      result.m[12] = a.m[12]*b.m[0] + a.m[13]*b.m[4] + a.m[14]*b.m[8] +  a.m[15]*b.m[12];
      result.m[13] = a.m[12]*b.m[1] + a.m[13]*b.m[5] + a.m[14]*b.m[9] +  a.m[15]*b.m[13];
      result.m[14] = a.m[12]*b.m[2] + a.m[13]*b.m[6] + a.m[14]*b.m[10] + a.m[15]*b.m[14];
      result.m[15] = a.m[12]*b.m[3] + a.m[13]*b.m[7] + a.m[14]*b.m[11] + a.m[15]*b.m[15];
   }

   inline void MatrixSetRotations4x4_YXZ(Matrix4x4& result, float yangle, float xangle, float zangle)
   {
      Matrix4x4 x;
      Matrix4x4 y;
      Matrix4x4 z;

      MatrixSetRotationX(x, xangle);
      MatrixSetRotationY(y, yangle);
      MatrixSetRotationZ(z, zangle);

      Matrix4x4 tmp;

      MatrixMultiply4x4(tmp, x, y);
      MatrixMultiply4x4(result, z, tmp);
   }

   inline void MatrixSetRotations4x4_XYZ(Matrix4x4& result, float xangle, float yangle, float zangle)
   {
      Matrix4x4 x;
      Matrix4x4 y;
      Matrix4x4 z;

      MatrixSetRotationX(x, xangle);
      MatrixSetRotationY(y, yangle);
      MatrixSetRotationZ(z, zangle);

      Matrix4x4 tmp;

      MatrixMultiply4x4(tmp, y, x);
      MatrixMultiply4x4(result, z, tmp);
   }

   inline void MatrixSetRotations4x4_XYZ_2(Matrix4x4& result, float xangle, float yangle, float zangle)
   {
      Matrix4x4 x;
      Matrix4x4 y;
      Matrix4x4 z;

      MatrixSetRotationX(x, xangle);
      MatrixSetRotationY(y, yangle);
      MatrixSetRotationZ(z, zangle);

      Matrix4x4 tmp;

      MatrixMultiply4x4(tmp, z, y);
      MatrixMultiply4x4(result, x, tmp);
   }

   inline void GetInverseRotationMatrix( Matrix4x4& result, vec3& direction, bool leftHanded = false, float ydir = 1.f )
   {
	   vec3 up( 0, ydir, 0 );
 	   up.Normalize();
 	   vec3 z = -direction;
	   z.Normalize();
 	   vec3 right = up.Cross(z);
 	   right.Normalize();
 	   vec3 u = z.Cross(right);
 	   u.Normalize();
 
 	   result.Identity();
 	   result.SetForward( leftHanded ? -z : z );
 	   result.SetRight(right);
 	   result.SetUp(u);
 	   result.Transpose();
   }

   struct Quat
   {
      Quat()
      {
         x = y = z = 0;
         w = 1;
      }

      Quat(float _x, float _y, float _z, float _w)
      {
         x = _x;
         y = _y;
         z = _z;
         w = _w;
      }

      Quat(vec4& v)
      {
         x = v.x;
         y = v.y;
         z = v.z;
         w = v.w;
      }

      float x,y,z,w;

      void Normalize()
      {
         float mag2 = w*w + x*x + y*y + z*z;
         float mag = sqrt(mag2);
         w /= mag;
         x /= mag;
         y /= mag;
         z /= mag;
      }


      void ToMatrix(Matrix4x4& mat)
      {
         float x2 = x * x;
         float y2 = y * y;
         float z2 = z * z;
         float xy = x * y;
         float xz = x * z;
         float yz = y * z;
         float wx = w * x;
         float wy = w * y;
         float wz = w * z;

         mat.m[0] = 1.0f - 2.0f * (y2 + z2); 
         mat.m[4] = 2.0f * (xy - wz); 
         mat.m[8] = 2.0f * (xz + wy); 
         mat.m[12] = 0.0f; 

         mat.m[1]  = 2.0f * (xy + wz); 
         mat.m[5]  = 1.0f - 2.0f * (x2 + z2); 
         mat.m[9]  = 2.0f * (yz - wx); 
         mat.m[13] = 0.0f; 

         mat.m[2]  = 2.0f * (xz - wy); 
         mat.m[6]  = 2.0f * (yz + wx); 
         mat.m[10] = 1.0f - 2.0f * (x2 + y2); 
         mat.m[14] = 0.0f; 

         mat.m[3]  = 0.f; 
         mat.m[7]  = 0.f; 
         mat.m[11] = 0.f; 
         mat.m[15] = 1.f;    

         // opengl column major
         //       return Matrix4( 1.0f - 2.0f * (y2 + z2), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
         //                       2.0f * (xy + wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz - wx), 0.0f,
         //                       2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (x2 + y2), 0.0f,
         //                         0.0f, 0.0f, 0.0f, 1.0f)
      }

      Quat Conjugate()
      {
         return Quat(-x, -y, -z, w);
      }

      Quat Quat::operator*(Quat& rq)
      {
         return Quat(w * rq.x + x * rq.w + y * rq.z - z * rq.y,
                     w * rq.y + y * rq.w + z * rq.x - x * rq.z,
                     w * rq.z + z * rq.w + x * rq.y - y * rq.x,
                     w * rq.w - x * rq.x - y * rq.y - z * rq.z);
      }

      vec3 operator* (vec3& vec)
      {
         vec3 vn(vec);

         vn.Normalize();

         Quat vecQuat, resQuat;
         vecQuat.x = vn.x;
         vecQuat.y = vn.y;
         vecQuat.z = vn.z;
         vecQuat.w = 0.0f;

         resQuat = vecQuat * Conjugate();
         resQuat = *this * resQuat;

         return( vec3(resQuat.x, resQuat.y, resQuat.z) );
      }

      void Quat::FromAxis(vec3& v, float angle)
      {
         float sinAngle;
         angle *= 0.5f;
         vec3 vn(v);
         vn.Normalize();

         sinAngle = sin( kDegreesToRadian(angle) );

         x = (vn.x * sinAngle);
         y = (vn.y * sinAngle);
         z = (vn.z * sinAngle);
         w = cos( kDegreesToRadian(angle) );
      }

      void Quat::FromEuler(float pitch, float yaw, float roll)
      {
         float p  = kDegreesToRadian(pitch);
         float yw = kDegreesToRadian(yaw);
         float r  = kDegreesToRadian(roll);

         float sinp = sin(p);
         float siny = sin(yw);
         float sinr = sin(r);
         float cosp = cos(p);
         float cosy = cos(yw);
         float cosr = cos(r);

         this->x = sinr * cosp * cosy - cosr * sinp * siny;
         this->y = cosr * sinp * cosy + sinr * cosp * siny;
         this->z = cosr * cosp * siny - sinr * sinp * cosy;
         this->w = cosr * cosp * cosy + sinr * sinp * siny;

         Normalize();
      }
   };

   inline void SetProjectionMatrix(Matrix4x4& result, float fov, float aspect, float nearPlane, float farPlane )
   {
	   float yscale = 1.f/tan( fov/2.f );
	   float xscale = yscale/aspect;

	   memset( result.m, 0x00, sizeof(float)*16 );

	   result.m[0]  = xscale;
	   result.m[5]  = yscale;
	   result.m[10] = farPlane/(farPlane-nearPlane);
	   result.m[11] = 1.f;
	   result.m[14] = -nearPlane*farPlane/(farPlane-nearPlane);

	   result.m[0]  = xscale;
	   result.m[5]  = yscale;
	   result.m[10] = farPlane/(farPlane-nearPlane);
	   result.m[11] = -nearPlane*farPlane/(farPlane-nearPlane);
	   result.m[14] = 1;

	   /*  xScale     0          0               0
		   0        yScale       0               0
		   0          0       zf/(zf-zn)         -zn*zf/(zf-zn)
		   0          0         1                 0 */
   }

   inline void SetOrthographicMatrix(Matrix4x4& result, float width, float height, float nearPlane, float farPlane )
   {
	   memset( result.m, 0x00, sizeof(float)*16 );

	   result.m[0]  = 2.f/width;
	   result.m[5]  = 2.f/height;
	   result.m[10] = 1.f/(farPlane-nearPlane);
	   result.m[14] = nearPlane/(nearPlane-farPlane);
	   result.m[15] = 1.f;

// 	   2/w  0    0           0
// 		   0    2/h  0           0
// 		   0    0    1/(zf-zn)   0
// 		   0    0    zn/(zn-zf)  1

   }
 
   inline void SetOrthographicOffCenterMatrix(Matrix4x4& result, float left, float right, float bottom, float top, float nearPlane, float farPlane )
   {
	   memset( result.m, 0x00, sizeof(float)*16 );

	   result.m[0]  = 2.f/(right-left);
	   result.m[5]  = 2.f/(top-bottom);
	   result.m[10] = 1.f/(farPlane-nearPlane);
	   result.m[12] = (left+right)/(left-right);
	   result.m[13] = (top+bottom)/(bottom-top);
	   result.m[14] = nearPlane/(nearPlane-farPlane);
	   result.m[15] = 1.f;

	   // 	   2/(r-l)      0            0           0
	   // 		   0            2/(t-b)      0           0
	   // 		   0            0            1/(zf-zn)   0
	   // 		   (l+r)/(l-r)  (t+b)/(b-t)  zn/(zn-zf)  1
   }

   inline void SetCameraWorldMatrix(Matrix4x4& result, vec3& eye, vec3& at, vec3& up)
   {
      vec3 z = at - eye;
      z.Normalize();
      vec3 right = z.Cross(up);
      right.Normalize();
      vec3 u = right.Cross(z);
      u.Normalize();
     
      result.Identity();
      result.SetForward(-z);
      result.SetRight(right);
      result.SetUp(u);
      result.SetTrans(eye);
   }

   inline void SetLookAtMatrix(Matrix4x4& result, vec3& eye, vec3& at, vec3& up)
   {
      vec3 z = at - eye;
      z.Normalize();
      vec3 right = up.Cross(z);
      right.Normalize();
      vec3 u = z.Cross(right);
      u.Normalize();

      Matrix4x4 rotmat;
      rotmat.Identity();
      rotmat.SetForward(z);
      rotmat.SetRight(right);
      rotmat.SetUp(u);
      rotmat.Transpose();

      Matrix4x4 transmat;
      transmat.Identity();
      transmat.SetTrans(-eye);

      result = rotmat * transmat;
   }

   inline bool FindInverse4x4(Matrix4x4& result, Matrix4x4& mat)
   {
      float inv[16];
      float det;
      int i;

      float* m = &mat.m[0];

      inv[0] =   m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
      + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
      inv[4] =  -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
      - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
      inv[8] =   m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
      + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
      inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
      - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
      inv[1] =  -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
      - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
      inv[5] =   m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
      + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
      inv[9] =  -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
      - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
      inv[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
      + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
      inv[2] =   m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
      + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
      inv[6] =  -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
      - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
      inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
      + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
      inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
      - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
      inv[3] =  -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
      - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
      inv[7] =   m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
      + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
      inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
      - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
      inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
      + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

      det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
      if (det == 0)
         return false;

      det = 1.f / det;

      for (i = 0; i < 16; i++)
         result.m[i] = inv[i] * det;

      return true;
   }

   inline void GetMatrix4x4(Matrix4x4& outMat, vec3& pos, vec3& rot)
   {
 //     MatrixSetRotations4x4_YXZ(outMat, rot.y, rot.x, rot.z);
      MatrixSetRotations4x4_XYZ(outMat, rot.x, rot.y, rot.z);
      outMat.SetTrans(pos);
   }

   inline void GetMatrix4x4FromQuat(Matrix4x4& outMat, vec3& pos, vec3& rot)
   {
      Quat q;
      q.FromEuler(rot.x, rot.y, rot.z);
      q.ToMatrix(outMat);
      outMat.SetTrans(pos);
   }

   inline bool FindInverse3x3(Matrix3x3& result, Matrix3x3& mat)
   {
      float invDet = 1.f / ( (mat.m[0]*mat.m[4]*mat.m[8]) + 
         (mat.m[1]*mat.m[5]*mat.m[6]) + 
         (mat.m[2]*mat.m[3]*mat.m[7]) - 
         (mat.m[0]*mat.m[5]*mat.m[7]) -
         (mat.m[1]*mat.m[3]*mat.m[8]) -
         (mat.m[2]*mat.m[4]*mat.m[6]) );

      result.m[0] = mat.m[4]*mat.m[8] - mat.m[7]*mat.m[5];
      result.m[1] = mat.m[2]*mat.m[7] - mat.m[1]*mat.m[8];
      result.m[2] = mat.m[1]*mat.m[5] - mat.m[2]*mat.m[4];

      result.m[3] = mat.m[5]*mat.m[6] - mat.m[3]*mat.m[8];
      result.m[4] = mat.m[0]*mat.m[8] - mat.m[2]*mat.m[6];
      result.m[5] = mat.m[2]*mat.m[3] - mat.m[5]*mat.m[0];

      result.m[6] = mat.m[3]*mat.m[7] - mat.m[4]*mat.m[6];
      result.m[7] = mat.m[1]*mat.m[6] - mat.m[0]*mat.m[7];
      result.m[8] = mat.m[0]*mat.m[4] - mat.m[3]*mat.m[1];

      for(int i=0; i<16; i++)
      {
         result.m[i] *= invDet;
      }

      return true;
   }

   inline bool FindInverse2x2(Matrix2x2& result, Matrix2x2& mat)
   {
      float invDet = 1.f / ( (mat.m[0]*mat.m[3]) - (mat.m[1]*mat.m[2]) );

      result.m[0] =  mat.m[0]*invDet;
      result.m[1] = -mat.m[1]*invDet;
      result.m[2] = -mat.m[2]*invDet;
      result.m[3] =  mat.m[0]*invDet;
   }

   inline vec3 reflect(vec3 v, vec3 normal)
   {
      float vdotn = v.Dot(normal);
      vec3 twoN = normal*vdotn*2.f;
      vec3 r = v + -twoN;
      return r;
   }

   inline float dot(vec3& a, vec3& b)
   {
      return a.Dot(b);
   }

   inline vec3 cross(vec3& a, vec3& b)
   {
      return a.Cross(b);
   }

   inline vec3 normalize(vec3& n)
   {
      vec3 v = n;
      v.Normalize();
      return v;
   }

   struct Triangle
   {
      vec3 p[3];

      void Set(vec3& p0, vec3& p1, vec3& p2)
      {
         p[0] = p0; p[1] = p1; p[2] = p2;
      }

      void GetBarycentric(vec3& outPnt, vec3& inPnt)
      {
         Matrix2x2 mat;
         Matrix2x2 invMat;

         mat.m[0] = p[0].x - p[2].x;
         mat.m[1] = p[1].x - p[2].x;
         mat.m[2] = p[0].y - p[2].y;
         mat.m[3] = p[1].y - p[2].y;

         FindInverse2x2(invMat,mat);

         vec3 barycentricCoords;
         Matrix2x2MultiplyVec2(barycentricCoords, invMat, inPnt);

         outPnt = barycentricCoords;
      }
   };

   struct Tetrahedron
   {
      vec3 p[4];

      void Set(vec3& p0, vec3& p1, vec3& p2, vec3& p3)
      {
         p[0] = p0; p[1] = p1; p[2] = p2; p[3] = p3;
      }

      void GetBarycentric(vec3& outPnt, vec3& inPnt)
      {
         Matrix3x3 T;
         Matrix3x3 invT;

         vec3 p3toP = inPnt - p[3];

         T.m[0] = p[0].x - p[3].x;	T.m[1] = p[1].x - p[3].x;	T.m[2]  = p[2].x - p[3].x;
         T.m[3] = p[0].y - p[3].y;	T.m[4] = p[1].y - p[3].y;	T.m[5]  = p[2].y - p[3].y;
         T.m[6] = p[0].z - p[3].z;	T.m[7] = p[1].z - p[3].z;	T.m[8]  = p[2].z - p[3].z;

         FindInverse3x3(invT, T);

         vec3 barycentricCoords;

         Matrix3x3MultiplyVec3(barycentricCoords, invT, inPnt);

         outPnt = barycentricCoords;
      }

   };

   enum VertexLayoutType
   {
      kVertLayout_PNT = 0,
      kVertLayout_PNTTB,
      kVertLayout_P,
      kVertLayout_PC,
      kVertLayout_PT,
      kVertLayout_Particle,
      kVertLayout_strand,
      kMaxVertLayoutTypes
   };

   struct Vertex_Particle
   {
      float pos[3];
      float uv[2];
      float color[4];
      float params[4];
   };

   struct Vertex_PNT
   {
      float pos[3];
      float norm[3];
      float uv[2];
   };

   struct Vertex_PNTTB
   {
      float pos[3];
      float norm[3];
      float uv[2];
      float tangent[3];
      float binormal[3];
   };

   struct Vertex_PNTTB_WithSkin
   {
      float pos[3];
      float norm[3];
      float uv[2];
      float tangent[3];
      float binormal[3];
      int     boneIndices[4];
      float boneWeights[4];
   };

   struct Vertex_P
   {
      float pos[3];
   };

   struct Vertex_PC
   {
      float pos[3];
      float color[4];
   };

   struct Vertex_PT
   {
      float pos[3];
      float uv[2];
   };

   struct Vertex_strand
   {
      float pos[3];
      float dist;
      float normal[3];
      float tangent[3];
      int     instID;
   };


   enum PlaneTestResult
   {
      kInside = 0,       
      kStraddle,
	  kOutside   
   };

   struct Sphere
   {
      float pos[4];
   };

   struct BoundingBox
   {
      BoundingBox()
      {
         Clear();
      }

	  BoundingBox( bool setInfinite )
	  {
		  if( setInfinite ) SetInfinite();
		  else	  		    Clear();
	  }

	  void Clear()
	  {
		  minVert[0] = minVert[1] = minVert[2] =  FLT_MAX;
		  maxVert[0] = maxVert[1] = maxVert[2] = -FLT_MAX;
	  }

	  void SetInfinite()
	  {
		  minVert[0] = minVert[1] = minVert[2] = -FLT_MAX;
		  maxVert[0] = maxVert[1] = maxVert[2] = FLT_MAX;
	  }

	  vec3 GetCentroid()
	  {
		vec3 pmin = vec3( minVert[0], minVert[1], minVert[2] );
		vec3 pmax = vec3( maxVert[0], maxVert[1], maxVert[2] );

		vec3 result = (pmin + pmax) * 0.5f;

		return result;
	  }

      void AddGeometry(int numVerts, Vertex_PNTTB* verts);
	  void AddPositions(int numVerts, vec3* verts);

      void Concat(BoundingBox* otherBB)
      {
         if( minVert[0] < otherBB->minVert[0] )
            minVert[0] = otherBB->minVert[0];
         if( minVert[1] < otherBB->minVert[1] )
            minVert[1] = otherBB->minVert[1];
         if( minVert[2] < otherBB->minVert[2] )
            minVert[2] = otherBB->minVert[2];

         if( maxVert[0] > otherBB->maxVert[0] )
            maxVert[0] = otherBB->maxVert[0];
         if( maxVert[1] > otherBB->maxVert[1] )
            maxVert[1] = otherBB->maxVert[1];
         if( maxVert[2] > otherBB->maxVert[2] )
            maxVert[2] = otherBB->maxVert[2];
      }

      bool Overlaps(BoundingBox& b);

	  void GetTranslatedBB( Matrix4x4& world, BoundingBox& translatedBox )
	  {
		vec3 v[2];
		
		v[0] = vec3( minVert[0], minVert[1], minVert[2] );
		v[1] = vec3( maxVert[0], maxVert[1], maxVert[2] );  

		Matrix4x4MultVec3( v[0], world, v[0] );
		Matrix4x4MultVec3( v[1], world, v[1] );
		
		translatedBox.Clear();

		translatedBox.AddPositions( 2, v );
	  }

	  float CalcRadius()
	  {
		  vec3 pmin = vec3( minVert[0], minVert[1], minVert[2] );
		  vec3 pmax = vec3( maxVert[0], maxVert[1], maxVert[2] );

		  vec3 c = GetCentroid();

		  float radius = (pmax - c).Length();

		  return radius;
	  }

	  // these structures float[4] so it can be passed easily to gpu
      float minVert[4];
      float maxVert[4];
   };

   struct Plane
   {
      Plane() {};
 
	  Plane ( vec3& p0, vec3& p1, vec3& p2 )
      {
         vec3 v0 = p1 - p0;
         vec3 v1 = p2 - p0;
         vec3 n  = v1.Cross( v0 );
         n.Normalize();

         normal = n;
         d      = normal.Dot(p0);
      }

     Plane( vec3& _normal, vec3& p )
      {
         d = normal.Dot( p );
      }

       Plane ( const Plane& plane )
       {
          normal = plane.normal;
          d      = plane.d;
       }

      Plane( float a, float b, float c, float _d )
      {
         normal = vec3( a, b, c );
         d      = -_d;
      } 

      vec4 AsVec4()
      {
         return vec4( normal.x, normal.y, normal.z, d );
      }

      float DistTo( vec3 &p )
      {
         float t = normal.Dot( p );
         t -= d;
         return t;
      }

      void Plane::Normalize()
      {
         float len = normal.Length();
         if( len <= 0.f ) return;

         normal = normal * (1.f/len );
         d /= len;
      }

      int WhichSide( vec3 &p )
      {
         float dist = DistTo( p );
         if ( dist < 0.0f ) return kOutside;
         if ( dist > 0.0f ) return kInside;
         return kStraddle;
      }

      float RayIntersect( vec3 origin, vec3 direction )
      {
         // ray plane intersection for plane equation form:  ax + by + cz + d = 0
         //  is  -( N.O + D ) / N.D.  Note that the SuPlane class has m_fContant == -d
         float t = ( d - normal.Dot( origin ) ) / normal.Dot( direction );
         return t;
      }
	  

      vec3    normal;      
      float d; 
   };

   inline vec3 Float3ToVec3( float* f)
   {
      return vec3( f[0], f[1], f[2] );
   }

   struct Frustum
   {
      enum FrustumPlane
      {
         kFrustumLeft = 0, 
         kFrustumRight,
         kFrustumTop,
         kFrustumBottom,
         kFrustumNear,  
         kFrustumFar      
      };

      Frustum()
      {
         planes[kFrustumLeft]    = Plane(-1,0,0,-1);
         planes[kFrustumRight]   = Plane(1,0,0,-1);
         planes[kFrustumTop]     = Plane(0,1,0,-1);
         planes[kFrustumBottom]  = Plane(0,-1,0,-1);
         planes[kFrustumNear]    = Plane(0,0,-1,0);
         planes[kFrustumFar]     = Plane(0,0,1,-1);
      }

      Frustum( Plane _planes[6] )
      {
         planes[0] = _planes[0]; planes[1] = _planes[1]; planes[2] = _planes[2];
         planes[3] = _planes[3]; planes[4] = _planes[4]; planes[5] = _planes[5];
      }

      void SetFromMatrixD3D( Matrix4x4& wvp, bool useD3D = true )
      {
         // This math is described in the Akenine-Moeller/Haines real-time rendering book, page 613-614
         // modified to work with D3D-style projection matrices

         // here is the math:
         // left plane     = -(m3 + m0)
         // right plane    = -(m3 - m0)
         // bottom plane   = -(m3 + m1)
         // top plane      = -(m3 - m1)
         // near plane     = -(m2)
         // far plane      = -(m3 - m2)

         vec4 M0 = vec4( wvp.m[0], wvp.m[1], wvp.m[2], wvp.m[3] );
         vec4 M1 = vec4( wvp.m[4], wvp.m[5], wvp.m[6], wvp.m[7] );
         vec4 M2 = vec4( wvp.m[8], wvp.m[9], wvp.m[10], wvp.m[11] );
         vec4 M3 = vec4( -wvp.m[12], -wvp.m[13], -wvp.m[14], -wvp.m[15] );

         vec4 tmp = M3 - M0;
         planes[kFrustumLeft]   = Plane( tmp.x, tmp.y, tmp.z, tmp.w );

         tmp = M3 + M0;
         planes[kFrustumRight]  = Plane( tmp.x, tmp.y, tmp.z, tmp.w );

         tmp = M3 - M1;
         planes[kFrustumBottom] = Plane( tmp.x, tmp.y, tmp.z, tmp.w );

         tmp = M3 + M1;
         planes[kFrustumTop]    = Plane( tmp.x, tmp.y, tmp.z, tmp.w );

         if( useD3D )
         {
            tmp = -M2;
            planes[kFrustumNear]   = Plane( tmp.x, tmp.y, tmp.z, tmp.w );
         }
         else
         {
            tmp = M3 - M2;
            planes[kFrustumNear]   = Plane( tmp.x, tmp.y, tmp.z, tmp.w );
         }

         tmp = M3 + M2;
         planes[kFrustumFar]    = Plane( tmp.x, tmp.y, tmp.z, tmp.w );

         // normalize the frustum planes so that point-plane dot products give true differences
         // this is important for bounding sphere tests to work properly
         for( int i=0; i<6; i++ )
         {
            planes[i].Normalize();
         }
      }

      int CullAxisAlignedBox( vec3& minpt, vec3& maxpt )
      {
         bool intersected = false;
         for( int j=0; j<6; j++ )
         {
            // figure out which BB diagonal is most appropriate to test against this plane
            float vmin[3];
            float vmax[3];

            vec3& normal = planes[j].normal;

            for( int i=0; i<3; i++ )
            {
               if( normal[i] >= 0 )
               {
                  vmin[i] = minpt[i];
                  vmax[i] = maxpt[i];
               }
               else
               {
                  vmin[i] = maxpt[i];
                  vmax[i] = minpt[i];
               }
            }

            if( planes[j].DistTo( vec3(vmin[0], vmin[1], vmin[2]) ) > 0 ) 
            {
               return kOutside;
            }
            if( !intersected && planes[j].DistTo( vec3(vmax[0], vmax[1], vmax[2]) ) >= 0 )
            {
               intersected = true;
            }
         }

         return ( intersected ) ? kStraddle : kInside;
      }

      int CullAxisAlignedBox( BoundingBox& bb)
      {
         return CullAxisAlignedBox( vec3(bb.minVert[0], bb.minVert[1], bb.minVert[2]), 
            vec3(bb.maxVert[0], bb.maxVert[1], bb.maxVert[2]) );
      }

      int CullSphere( vec3& center, float radius )
      {
         bool intersected = false;
         for( int i=0; i<6; i++ )
         {
            float dist = planes[i].DistTo( center );
            if( dist > radius )
            {
               // entire sphere is outside one of the six planes, cull immediately
               return kOutside;
            }

            // sphere is either intersecting this plane, or all the way on the negative side

            // check if it intersects the plane
            if( dist > -radius )
            {
               intersected = true;
            }
         }
         return ( intersected ) ? kStraddle : kInside;
      }

      int CullSphere( Sphere& sphere )
      {
         vec3 center = vec3( sphere.pos[0], sphere.pos[1], sphere.pos[2] );
         return CullSphere( center, sphere.pos[3] );
      }

      Plane& GetPlane( int i ) { return planes[i]; }; 

      Plane planes[6];
   };

   vec3 ToneMap(vec3& v);

 };

   

   using namespace kiMath;

#endif
