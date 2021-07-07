#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda.h>
#include "cuda_runtime.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

#define PI 3.14159265
#define row 128
#define col 64

#define nbin 9
#define cell_x 16
#define cell_y 16

#define block_x 2
#define block_y 2

__global__
void kernel(float *d_mag, float *d_ang, float *d_output, int columns)
{
  //3200
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // id = 0,
  double bin[nbin] = {0};
  __syncthreads();

  int temp = columns/cell_x;    //temp = 40
  int i,j;
  i = (int)(id/temp) * cell_y;
  j = (int)(id%temp) * cell_x;
  i = i*columns + j;

  for(int b = 0; b < cell_y; b++)
  {
    for(int a = i; a < i+cell_x; a++)
    {
      double angle, px;
      int bin_size = 180/nbin;
      int ind, ind1;
      double val;
      px = d_mag[a+(b*columns)];
      angle = d_ang[a+(b*columns)];

      if(angle >= 180)
        angle = angle-180;

      ind = angle/bin_size;
      ind1 = (ind+1)%nbin;
      val = px*((bin_size-(angle-(ind*bin_size)))/bin_size);
      bin[ind] += val;
      bin[ind1] += px-val;
    }
  }
  // if(id == 0)
  // {
  //   printf("ID: %d, threadId: %d\n", id, threadIdx.x);
  //     for(int z = 0; z < nbin; z++)
  //     printf("bin[%d]: %f\n", z, bin[z]);
  // }
  __syncthreads();

  int l = (int)(id * nbin);
  for(int k = 0; k < nbin; k++)
    d_output[k+l] = bin[k];
}

__global__
void block_norm(int lim, int n, int x, float* hog, float* hogVecSq, float* res)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(id < lim)
  {
    float blockSqSum = 0;
    blockSqSum = hogVecSq[id] + hogVecSq[id+1] + hogVecSq[id+40] + hogVecSq[id+41];

    float s = sqrt(blockSqSum);
    int blocksize = nbin*block_x*block_y; //36

    for(int a = 0; a < blocksize; a++)
    {
      int pos;
      if(a < blocksize/2)
        pos = a + (id * nbin);
      else
        pos = a + (id * nbin) + x;

      res[a + id*blocksize] = (float)(hog[pos]/s);
    }
  // __syncthreads();
  //   if(id == 100)
  //   for(int i = 0; i < 100; i++)
  //     printf("%f\n", res[i]);
  }
}

double cosine_similarity(float* A, vector<double> B, unsigned int vector_Length)
{
    double dot = 0.0, a = 0.0, b = 0.0 ;
     for(unsigned int i = 0; i < vector_Length; ++i) {
        dot += A[i] * B[i] ;
        a += A[i] * A[i] ;
        b += B[i] * B[i] ;
    }
    return dot / (sqrt(a) * sqrt(b)) ;
}

float* calc_hog(Mat img, int res_size)
{
    float* res = (float*)calloc(res_size, sizeof(float));
    cvtColor(img,img,COLOR_BGR2GRAY);

    img.convertTo(img,CV_32F,1/255.0);

    //Calculate gradients in x and y direction
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    //Calculate gradient mag. and direction
    Mat mag, ang;
    cartToPolar(gx, gy, mag, ang, 1);

    int ncells = mag.cols/cell_x * mag.rows/cell_y; //3200

    // 640 x 1280 - 16 x 16 - 3200
    dim3 block (cell_x);
    dim3 threadsPerBlock (ncells/cell_x); // 3200/16 - 200

    const int inputSize = mag.cols * mag.rows * sizeof(float);
    const int outputSize = nbin * ncells * sizeof(float);

    float *d_mag, *d_ang, *d_output;
    float* hogVec = (float*)malloc(outputSize);

    cudaMalloc<float>(&d_mag, inputSize);
    cudaMalloc<float>(&d_ang, inputSize);
    cudaMalloc<float>(&d_output, outputSize);

    // Copy data from OpenCV input image to device memory
    cudaMemcpy(d_mag, mag.ptr(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ang, ang.ptr(), inputSize, cudaMemcpyHostToDevice);

    kernel <<<block,threadsPerBlock>>> (d_mag, d_ang, d_output, mag.cols);

    cudaDeviceSynchronize();
    cudaMemcpy(hogVec, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_mag);
    cudaFree(d_ang);
    cudaFree(d_output);

    int ncell_blocks = ((mag.cols/cell_x) - block_x + 1) * ((mag.rows/cell_y) - block_y + 1);
    int ncell_x = mag.cols/cell_x;

    float* hogVecSq = (float*)malloc(ncells*sizeof(float));
    float* hogVec1 = (float*)malloc(outputSize);
    // hogVecsq[0] = a2 + b2 + .. , hogVecSq[0-3200]
    for(int i = 0; i < (nbin * ncells); i++)
      hogVec1[i] = hogVec[i]*hogVec[i];

    for(int i = 0; i < ncells; i++)
    {
      float temp = 0;
      int pos = i*nbin;
      for(int j = 0; j < nbin; j++)
      {
        temp += hogVec1[j + pos];
      }
      hogVecSq[i] = temp;
    }

    float *d_hogVec, *d_hogVecSq, *d_res;

    cudaMalloc<float>(&d_hogVec, inputSize);
    cudaMalloc<float>(&d_hogVecSq, inputSize);
    cudaMalloc<float>(&d_res, res_size*sizeof(float));

    cudaMemcpy(d_hogVec, hogVec, outputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hogVecSq, hogVecSq, ncells*sizeof(float), cudaMemcpyHostToDevice);

    block_norm <<<block, threadsPerBlock>>> (ncell_blocks, ncells, ncell_x, d_hogVec, d_hogVecSq, d_res);

    cudaDeviceSynchronize();

    cudaMemcpy(res, d_res, res_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_hogVec);
    cudaFree(d_hogVecSq);
    cudaFree(d_res);

    free(hogVec);
    free(hogVec1);
    free(hogVecSq);

    // for(int i = 0; i < 10; i++)
      // printf("%f\n", res[i]);
    return res;
}

int main(int argc,char **argv)
{

  Mat img;
  img = imread(argv[1], 1);
  int size = ((img.cols/cell_x) - block_x + 1) * ((img.rows/cell_y) - block_y + 1);
  size *= nbin * block_x * block_y;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Start time
  cudaEventRecord(start);
// ---------------------------
  // ofstream file;
  // file.open("feat_cuda.dat");
  // for(int i = 1; i <=400; i++)
  // {
  //   Mat img = imread("images/"+to_string(i)+".jpg");
  //   // cout << i <<"\n";
  //   float* fd = calc_hog(img,size);
  //   for(int i=0;i<size;i++)
  //     file<<fd[i]<<" ";
  //   file<<"\n";
  //   free(fd);
  // }
  // file.close();
// -----------------------------
  float* fd = calc_hog(img, size);

  // Stop time
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;

  // Calculate elapsed time in milisecond
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<< "\nProcessing time on GPU (ms): " << milliseconds << "\n";

  cout<<size<<"\n";

  ofstream file;
  file.open("fd.dat");
  for(int i=0;i<size;i++)
    file<<fd[i]<<"\n";
  file.close();
  free(fd);

  // Mat img;
  // img=imread(argv[1],1);
  // int size = ((img.cols/cell_x) - block_x + 1) * ((img.rows/cell_y) - block_y + 1);
  // size *= nbin * block_x * block_y;
  //
  // float* fd =calc_hog(img,size);
  // vector<string> feat;
  // ifstream f("feat_cuda.dat");
  // string str;
  // while(getline(f,str)){
  //   feat.push_back(str);
  // }
  //
  // double max_t=0;
  // int maxi;
  // for(int i=0;i<400;i++){
  //   vector<double> hf;
  //   stringstream ss(feat[i]);
  //   string w;
  //   while(ss>>w)
  //     hf.push_back(stod(w));
  //   double t=cosine_similarity(fd,hf,size);
  //   if(t>max_t){
  //     max_t=t;
  //     maxi=i;
  //   }
  //
  // }
  // cout<<maxi+1<<"\n"<<max_t<<"\n";
  return 0;
}
