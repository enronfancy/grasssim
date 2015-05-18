#include "mathutils.h"

namespace kiMath
{
	void BoundingBox::AddPositions( int numPos, vec3* pos )
	{
		for(int i=0; i<numPos; i++)
		{
			if( pos[i].x < minVert[0] )
			{
				minVert[0] = pos[i].x;
			}

			if( pos[i].y < minVert[1] )
			{
				minVert[1] = pos[i].y;
			}

			if( pos[i].z < minVert[2] )
			{
				minVert[2] = pos[i].z;
			}

			if( pos[i].x > maxVert[0] )
			{
				maxVert[0] = pos[i].x;
			}

			if( pos[i].y > maxVert[1] )
			{
				maxVert[1] = pos[i].y;
			}

			if( pos[i].z > maxVert[2] )
			{
				maxVert[2] = pos[i].z;
			}
		} 
	}

   void BoundingBox::AddGeometry(int numVerts, Vertex_PNTTB* verts)
   {
      for(int i=0; i<numVerts; i++)
      {
         vec3 pos = vec3( verts[i].pos[0], verts[i].pos[1], verts[i].pos[2] );
		 AddPositions( 1, &pos );
	  }
   }

   bool BoundingBox::Overlaps(BoundingBox& b)
   {
      if( b.minVert[0] > maxVert[0] ) return false;
      if( b.minVert[1] > maxVert[1] ) return false;
      if( b.minVert[2] > maxVert[2] ) return false;

      if( b.maxVert[0] < minVert[0] ) return false;
      if( b.maxVert[1] < minVert[1] ) return false;
      if( b.maxVert[2] < minVert[2] ) return false;

      return true;
   }

   vec3 Centroid( int numPoints, vec3* pnts )
   {
	   vec3 total(0,0,0);

	   float oneOverNumPoints = 1.f/(float)numPoints;
	   for( int i=0; i<numPoints; i++ )
	   {
		   total += pnts[i];
	   }

	   vec3 result = total * oneOverNumPoints;

	   return result;
   }


};