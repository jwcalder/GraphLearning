/* vector_operations.h - 
 *
 *  Basic vector calculus operations.
 *
 *  Author: Jeff Calder, 2018.
 */

#define ABS(a) (((a)<0)?-(a):(a))
#define SIGN(a) (((a)<0)?(-1):(1))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN3(a,b,c) (MIN(MIN(a,b),c))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MAX3(a,b,c) (MAX(MAX(a,b),c))
#define PI 3.14159265359

#define dot(x,y) (x[0]*y[0] + x[1]*y[1] + x[2]*y[2])
#define norm(x) (sqrt(x[0]*x[0] + x[1]*x[1] +  x[2]*x[2]))
#define norm_squared(x) (x[0]*x[0] + x[1]*x[1] +  x[2]*x[2])
#define dist(x,y) (sqrt((x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])))
#define dist_squared(x,y) ((x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2]))
#define cross(x,y,z) z[0] = x[1]*y[2] - x[2]*y[1]; z[1] = x[2]*y[0] - x[0]*y[2]; z[2] = x[0]*y[1] - x[1]*y[0]
#define centroid(x,y,z,p) p[0] = (x[0] + y[0] + z[0])/3; p[1] = (x[1] + y[1] + z[1])/3; p[2] = (x[2] + y[2] + z[2])/3
#define average(x,y,z) z[0] = (x[0] + y[0])/2; z[1] = (x[1] + y[1])/2; z[2] = (x[2] + y[2])/2
#define add(x,y,z) z[0] = x[0] + y[0]; z[1] = x[1] + y[1]; z[2] = x[2] + y[2]
#define sub(x,y,z) z[0] = x[0] - y[0]; z[1] = x[1] - y[1]; z[2] = x[2] - y[2]
#define mult(x,a,z) z[0] = a*x[0]; z[1] = a*x[1]; z[2] = a*x[2]
#define new_coordinates(x,a,b,c) v1 = dot(x,e1); v2 = dot(x,e2); v3 = dot(x,e3);x[0] = v1; x[1] = v2; x[2] = v3

