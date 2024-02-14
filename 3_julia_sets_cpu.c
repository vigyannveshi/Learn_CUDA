/*
A 32 bit per pixel color bitmap represents a 2D grid of pixel values where each pixel is represented by 4 channels (R,G,B,alpha) and where each channel has values in the range (0,255) (alpha represents transperancy)
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include "bitmap_library.h"

#define DIM 640

struct Complex {
    float r;
    float i;
};


float magnitude(struct Complex a){
    return (float) sqrt((a.r*a.r)+(a.i*a.i));
}

float magnitudeSquare(struct Complex a){
    return ((a.r*a.r)+(a.i*a.i));
}

void add(struct Complex a, struct Complex b,struct Complex *res){
    res->r=a.r+b.r;
    res->i=a.i+b.i;
}

void mul(struct Complex a, struct Complex b, struct Complex *res){
    res->r=(a.r*b.r)-(a.i*b.i);
    res->i=(a.i*b.r)+(a.r*b.i);
}

void disp(struct Complex *z){
    if (z->i>0){
        printf("%f+%fi",z->r,z->i);
    }
    else{
        printf("%f %fi",z->r,z->i);
    }
}

int julia(int x, int y){
    const float scale=1.5;
    float jx =scale*(float) (DIM/2 -x)/(DIM/2);
    float jy =scale*(float) (DIM/2 -y)/(DIM/2);

    struct Complex c,a,r1,r2;
    c.r=-0.8;
    c.i=0.154;

    a.r=jx;
    a.i=jy;

    int i=0;
    
    for(i=0; i<200;i++){
        // a=a*a+c;
        mul(a,a,&r1);
        add(r1,c,&r2);
        if (magnitudeSquare(r2)>1000){
            return 0; // return 0 if it is not in set
        }
        a.r=r2.r;
        a.i=r2.i;
    }
    return 1; // return 1 if the point is in the set 
}

void kernel(unsigned char *ptr)
{
    for (int y=0; y<DIM; y++){
        for(int x=0; x<DIM; x++){
            int offset=x+y*DIM;
            int juliaValue = julia(x,y,DIM);
            ptr[offset*4 + 0] = 255*juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}

int main(void){
    CPUBitmap bitmap (DIM,DIM);
    unsigned char *ptr =bitmap.get_ptr();
    kernel(ptr);
    bitmap.display_and_exit();
}

// int main(){
//     struct Complex a, b, result;

//     // Initialize complex numbers a and b
//     a.r = 5;
//     a.i = 7;
//     b.r = 3;
//     b.i = 2;

//     // Display complex numbers a and b
//     printf("a = ");
//     disp(&a);
//     printf("\n");

//     printf("b = ");
//     disp(&b);
//     printf("\n");

//     // Compute and display magnitude of a
//     printf("Magnitude of a: %f\n", magnitude(a));

//     // Add a and b, store result in result
//     add(a, b, &result);
//     printf("a + b = ");
//     disp(&result);
//     printf("\n");

//     // Multiply a and b, store result in result
//     mul(a, b, &result);
//     printf("a * b = ");
//     disp(&result);
//     printf("\n");

//     return 0;
// }

