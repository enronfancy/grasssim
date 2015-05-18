#ifndef _MY_FLOAT
#define _MY_FLOAT


struct float2
{
public:
	float x;
	float y;

	float2()
	{
		this->x = this->y  = 0.0f;
	}

	float2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}
	
	float2(const float2 &f)
	{
		this->x = f.x;
		this->y = f.y;
	}

	float2 operator+(const float2& f)
	{
		float newx = this->x + f.x;
		float newy = this->y + f.y;
	
		return float2(newx,newy);
	}

	float2 operator-(const float2& f)
	{
		float newx = this->x - f.x;
		float newy = this->y - f.y;
	
		return float2(newx,newy);
	}

	float2 operator*(const float2& f)
	{
		float newx = this->x * f.x;
		float newy = this->y * f.y;
	
		return float2(newx,newy);
	}
	
	float2 operator/(const float2& f)
	{
		if (f.x == 0 || f.y == 0)
			return float2(0.0f,0.0f);

		float newx = this->x / f.x;
		float newy = this->y / f.y;
	
		return float2(newx,newy);
	}

	float2& operator=(const float2& f)
	{
		this->x = f.x;
		this->y = f.y;
	
		return *this;
	}

};

struct float3
{
public:
	float x;
	float y;
	float z;
	
	float3()
	{
		this->x = this->y = this->z = 0.0f;
	}

	float3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}
	
	float3(const float3 &f)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = f.z;
	}

	float3(const float2 &f, float z)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = z;
	}

	float3 operator+(const float3& f)
	{
		float newx = this->x + f.x;
		float newy = this->y + f.y;
		float newz = this->z + f.z;
	
		return float3(newx,newy,newz);
	}

	float3 operator-(const float3& f)
	{
		float newx = this->x - f.x;
		float newy = this->y - f.y;
		float newz = this->z - f.z;
	
		return float3(newx,newy,newz);
	}

	float3 operator*(const float3& f)
	{
		float newx = this->x * f.x;
		float newy = this->y * f.y;
		float newz = this->z * f.z;
	
		return float3(newx,newy,newz);
	}
	
	float3 operator/(const float3& f)
	{
		if (f.x == 0 || f.y == 0 || f.z == 0)
			return float3(0.0f,0.0f,0.0f);

		float newx = this->x / f.x;
		float newy = this->y / f.y;
		float newz = this->z / f.z;
	
		return float3(newx,newy,newz);
	}

	float3& operator=(const float3& f)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = f.z;
	
		return *this;
	}
};

struct float4
{
public:
	float x;
	float y;
	float z;
	float w;

	
	float4()
	{
		this->x = this->y = this->z = this->w = 0.0f;
	}

	float4(float x, float y, float z, float w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
	
	float4(const float4 &f)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = f.z;
		this->w = f.w;
	}

	float4(const float3 &f, float w)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = f.z;
		this->w = w;
	}

	float4(const float2 &f, float z, float w)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = z;
		this->w = w;
	}


	float4 operator+(const float4& f)
	{
		float newx = this->x + f.x;
		float newy = this->y + f.y;
		float newz = this->z + f.z;
		float neww = this->w + f.w;
	
		return float4(newx,newy,newz, neww);
	}

	float4 operator-(const float4& f)
	{
		float newx = this->x - f.x;
		float newy = this->y - f.y;
		float newz = this->z - f.z;
		float neww = this->w - f.w;
	
		return float4(newx,newy,newz,neww);
	}

	float4 operator*(const float4& f)
	{
		float newx = this->x * f.x;
		float newy = this->y * f.y;
		float newz = this->z * f.z;
		float neww = this->w * f.w;
	
		return float4(newx,newy,newz,neww);
	}
	
	float4 operator/(const float4& f)
	{
		if (f.x == 0 || f.y == 0 || f.z == 0 || f.w == 0)
			return float4(0.0f,0.0f,0.0f,0.0f);

		float newx = this->x / f.x;
		float newy = this->y / f.y;
		float newz = this->z / f.z;
		float neww = this->w / f.w;
	
		return float4(newx,newy,newz,neww);
	}

	float4& operator=(const float4& f)
	{
		this->x = f.x;
		this->y = f.y;
		this->z = f.z;
		this->w = f.w;
	
		return *this;
	}


};


#endif