#include<iostream>
#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<cstring>
#include<stdlib.h>
#include<vector>
#include<algorithm>
#include<cmath>
#include<ctime>
#include<fstream>
#include<string>
#include<omp.h>
#include<chrono>
using namespace cv;
using namespace std;
using namespace std::chrono;
#define row 128
#define col 64


#define nbin 9
#define cell_x 32
#define cell_y 32

#define block_x 2
#define block_y 2

#define PI 3.14159265
int max_t,chunk,r,c,rem;

void grad_x(Mat& img,Mat& gx,int n){
  //int r=img.rows;
  //int c=img.cols;

  int i,j;
  float *p1,*p2;
  int l;
  if(n+1==max_t/2)
    l=r;
  else
    l=(n+1)*chunk;
  for(int i=n*chunk;i<l;++i){

    p1=img.ptr<float>(i);
    p2=gx.ptr<float>(i);
    p2[0]=0;
    p2[c-1]=0;
    for(int j=1;j<c-1;++j){
      p2[j]=p1[j+1]-p1[j-1];
      //cout<<p2[j]<<"\n";
    }
  }
}

void grad_y(Mat& img,Mat& gy,int n){
  //int r=img.rows;
  //int c=img.cols;
  int l;
  float* p1,*p2,*p3;
  if(n+1==max_t/2)
    l=r-1;
  else
    l=((n+1)*chunk)+1;
  for(int i=(n*chunk)+1;i<l;i++){
    p1=img.ptr<float>(i-1);
    p2=img.ptr<float>(i+1);
    p3=gy.ptr<float>(i);
    for(int j=0;j<c;j++){
      p3[j]=p2[j]-p1[j];
    }
  }
}


void polar(Mat& gx,Mat& gy,Mat& mag,Mat& ang,int n){
  //int r=gx.rows;
  //int c=gx.cols;
  float* p1,*p2,*p3,*p4;
  float angle;

  int l;
  if(n==max_t-1)
    l=r;
  else
    l=(n+1)*chunk;
  for(int i=n*chunk;i<l;i++){
    p1=gx.ptr<float>(i);
    p2=gy.ptr<float>(i);
    p3=mag.ptr<float>(i);
    p4=ang.ptr<float>(i);
    for(int j=0;j<c;j++){
      p3[j]=sqrt((p1[j]*p1[j])+(p2[j]*p2[j]));

    }
    for(int j=0;j<c;j++){
      angle=atan2(p2[j],p1[j])*180/PI;
      if(angle<0)
        p4[j]=angle+360;
      else
        p4[j]=angle;
    }
  }

}

void check_dim(Mat img){
  printf("%d",img.cols);
  if(!(img.rows % row ==0 && img.cols % col==0)){
    printf("Invalid size of image\n");
    exit(0);
  }
}

void print_pixels(Mat img){
  vector<float> image;
  int r=img.rows;
  int c=img.cols;
  image.assign((float*)img.datastart,(float*)img.dataend);
  for(int i=0;i<r*c;i++){
      cout<<image[i]<<"\n";
  }
}

void compare(Mat& a,Mat& b){
  int r=a.rows;
  int c=a.cols;
  float* p1,*p2;
  for(int i=0;i<r;i++){
    p1=a.ptr<float>(i);
    p2=b.ptr<float>(i);
    for(int j=0;j<c;j++){
      if(p1[j]!=p2[j])
      if(fabs(p1[j]-p2[j])>=0.01 )
        cout<<p1[j]-p2[j]<<"\n";
    }
  }
}


vector<double> getFeature(Mat cell,Mat ang){
  double angle,px;
  vector<double> bin(nbin,0);
  int bin_size=180/nbin;
  int ind,ind1;
  double val;
  for(int i=0;i<ang.rows;i++){

    for(int j=0;j<ang.cols;j++){
      px=cell.at<float>(i,j);
      angle=ang.at<float>(i,j);
      if(angle>=180)
        angle=angle-180;

      ind=angle/bin_size;
      ind1=(ind+1)%nbin;
      val=px*((bin_size-(angle-(ind*bin_size)))/bin_size);
      bin[ind]+=val;
      bin[ind1]+=px-val;
    }
  }
  return bin;
}

void block_norm(vector<vector<double>>& hog,vector<double>& sq,int x,int y,int n,vector<double>& norm){

  int nb=chunk;
  if(n<rem){
    nb+=1;
  }
  vector<double> hog_block;
  int start;
  if(n<=rem)
    start=n*(chunk+1);
  else
    start=rem*(chunk+1)+(n-rem)*chunk;

  for(int i=0;i<nb;i++){
    double s=0;
    int st=start+i;
    if(x-(st%x)<block_x)
      st+=block_x;
    for(int j=0;j<block_y;j++){
      int ind=j*x;
      for(int k=0;k<block_x;k++){
        s+=sq[st+ind+k];
      }
    }
    s=sqrt(s);
    for(int j=0;j<block_y;j++){
      int ind=j*x;
      for(int k=0;k<block_x;k++){
        for(int l=0;l<nbin;l++){
          if(s)
            //norm[st*36+ind+9*k+l]=hog[st+ind+9*k][l]/s;
            hog_block.push_back(hog[st+ind+k][l]/s);
          else
            //norm[st*36+ind+9*k+l]=0;
            hog_block.push_back(0);
        }

      }
    }
  }
  for(int i=0;i<hog_block.size();i++){
    norm[start*36+i]=hog_block[i];
  }

}

vector<double> calc_hog(Mat img){
  omp_set_num_threads(4);
    int cy=img.rows/cell_y;
    int cx=img.cols/cell_x;
    int size=(cx)*(cy);
    int norm_size=(cx-block_x+1)*(cy-block_y+1);
    vector<vector<double>> hogVec(size,vector<double>(9));
    vector<double> norm(norm_size*block_x*block_y*nbin);
    //vector<double> hogVec[size];
    vector<double> sq(size,0);
    cvtColor(img,img,COLOR_BGR2GRAY);

    img.convertTo(img,CV_32F,1/255.0);
    Mat gx,gy;
    Mat mag,ang;
    int cells;
    //float *p1,*p2,*p3,*p4;
    //cout<<1;
    //Calculate gradients in x and y direction

    //Sobel(img,gx,CV_32F,1,0,1);
    //Sobel(img,gy,CV_32F,0,1,1);

    #pragma omp parallel
    {

    #pragma omp single
    {
      max_t=omp_get_num_threads();
      //cout<<max_t;
      r=img.rows;
      c=img.cols;
      chunk=(int)r/(max_t/2);
      gx=Mat(r,c,CV_32FC1);
      gy=Mat(r,c,CV_32FC1);
      float *p1=gy.ptr<float>(0);
      float *p2=gy.ptr<float>(r-1);
      for(int i=0;i<c;i++){
        p1[i]=0;
        p2[i]=0;
      }
      mag=Mat(r,c,CV_32FC1);
      ang=Mat(r,c,CV_32FC1);
      cells=c/cell_x;
    }
    #pragma omp barrier
    int n=omp_get_thread_num();
    if(n%2==0){
      grad_x(img,gx,n/2);
    }
      //cout<<n/2<<"\n";
    else
      grad_y(img,gy,(n-1)/2);
      //cout<<(n-1)/2<<"\n";

    //Calculate gradient mag. and direction
    #pragma omp barrier
    //cartToPolar(gx,gy,mag,ang,1);
    #pragma omp single
    {
      chunk=(int)r/max_t;
    }
    polar(gx,gy,mag,ang,n);

    #pragma omp barrier

    #pragma omp single
    {
      chunk=(int)((r/cell_y)*(c/cell_x))/max_t;
    }
    /*
    for(int i=0;i<r;i+=cell_x){
      for(int j=0;j<c;j+=cell_y){
          vector<double> bin;
          Mat cell_mag=mag.rowRange(i,i+cell_x).colRange(j,j+cell_y);
          Mat cell_ang=ang.rowRange(i,i+cell_x).colRange(j,j+cell_y);
          bin=getFeature(cell_mag,cell_ang);
          hogVec.push_back(bin);
      }
    }*/
    int start=n*chunk;
    int start_y=((start)%cells)*cell_x;
    int start_x=((int)(start)/cells)*cell_y;
    //auto it=hogVec.begin()+start;
    //cout<<size;
    for(int i=0;i<chunk;i++){
      vector<double> bin;
      Mat cell_mag=mag.rowRange(start_x,start_x+cell_x).colRange(start_y,start_y+cell_y);
      Mat cell_ang=ang.rowRange(start_x,start_x+cell_x).colRange(start_y,start_y+cell_y);
      bin=getFeature(cell_mag,cell_ang);
      hogVec[start++]=bin;
      for(int k=0;k<nbin;k++)
        sq[start-1]+=bin[k]*bin[k];
      //if(start++>=size)
        //cout<<start-1<<"\n";
      //hogVec.push_back(bin);
      start_y+=cell_x;
      if(start_y>=c){
        start_y=0;
        start_x+=cell_y;
      }
    }
    #pragma omp single
    {
      chunk=(int)norm_size/max_t;
      rem=norm_size%max_t;
    }
    #pragma omp barrier
    block_norm(hogVec,sq,c/cell_x,r/cell_y,n,norm);
  }
    //cout<<hogVec.size()<<"\n";
    //print_pixels(ang);
    //return block_norm(hogVec,sq,mag.cols/cell_x,mag.rows/cell_y);

    return norm;
}

double cosine_similarity(vector<double> A, vector<double> B, unsigned int vector_Length)
{
    double dot = 0.0, a = 0.0, b = 0.0 ;
     for(unsigned int i = 0; i < vector_Length; ++i) {
        dot += A[i] * B[i] ;
        a += A[i] * A[i] ;
        b += B[i] * B[i] ;
    }
    return dot / (sqrt(a) * sqrt(b)) ;
}

int main(int argc,char **argv){
  /*
  Mat img;
  img=imread(argv[1],1);
  auto start = high_resolution_clock::now();
  vector<double> fd=calc_hog(img);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout<<duration.count()<<"\n";
  cout<<fd.size()<<"\n";

  ofstream file;
  file.open("fd1.dat");
  for(int i=0;i<fd.size();i++)
    file<<fd[i]<<"\n";
  file.close();*/

  Mat img;
  img=imread(argv[1],1);

  vector<double> fd=calc_hog(img);
  vector<string> feat;
  ifstream f("feat_omp.dat");
  string str;
  while(getline(f,str)){
    feat.push_back(str);
  }

  double max_t=0;
  int maxi;
  for(int i=0;i<400;i++){
    vector<double> hf;
    stringstream ss(feat[i]);
    string w;
    while(ss>>w)
      hf.push_back(stod(w));
    double t=cosine_similarity(fd,hf,fd.size());
    if(t>max_t){
      max_t=t;
      maxi=i;
    }

  }
  cout<<maxi+1<<"\n"<<max_t<<"\n";

  /*
  ofstream file;
  file.open("feat_omp.dat");
  auto start = high_resolution_clock::now();
  for(int i=1;i<=400;i++){
    Mat img=imread("images/"+to_string(i)+".jpg");
    cout<<i<<"\n";

    vector<double> fd=calc_hog(img);
    for(int i=0;i<fd.size();i++)
      file<<fd[i]<<" ";
    file<<"\n";
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout<<duration.count()<<"\n";
  file.close();*/


  return 0;
}
