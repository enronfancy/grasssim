#include "draw.h"
#include <GL/GLU.h>

void init(void)
{
  glClearColor(0.0,0.0,0.0,0.0);		//ָ�������ɫ����ɫ������������ɫ
  glShadeModel(GL_FLAT);				//������ɫģʽ�����ú㶨��ɫ				
}

void RenderScene()
{
    //�����ɫ��������������ɫ�� glClearColor( 0, 0.0, 0.0, 1 ); ָ��Ϊ��ɫ

    glClear( GL_COLOR_BUFFER_BIT );
    glPointSize( 9 );//ָ����Ĵ�С��9�����ص�λ
    glColor3f( 1.0f, 0.0f, 0.0f );//ָ�������ɫ
    glBegin( GL_POINTS );//��ʼ����
    {
        glVertex3f(0.0f, 0.0f, 0.0f); // ������Ϊ(0,0,0)�ĵط�������һ����
    }
    glEnd();//��������
    glutSwapBuffers();
}

void SetupRC()
{
    glClearColor( 0.0f, 0.0f, 0.0f, 1 );//��RGB(0,0,0)����ɫ�������ɫ������

    glColor3f( 1.0f, 0.0f, 0.0f );//��RGB(1,0,0)����ɫ������ͼ��

}


void ChangeSize( GLsizei w, GLsizei h )
{
    GLfloat nRange = 200.0f;

    // Prevent a divide by zero

    if(h == 0)
        h = 1;

    // Set Viewport to window dimensions

    glViewport(0, 0, w, h);

    // Reset projection matrix stack

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Establish clipping volume (left, right, bottom, top, near, far)

    if (w <= h) 
        glOrtho (-nRange, nRange, -nRange*h/w, nRange*h/w, -nRange, nRange);
    else 
        glOrtho (-nRange*w/h, nRange*w/h, -nRange, nRange, -nRange, nRange);

    // Reset Model view matrix stack

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
