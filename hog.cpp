#include<iostream>
#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<cstring>
#include<stdlib.h>
#include<vector>
#include<algorithm>
#include<cmath>
#include<fstream>
#include<string>
#include<chrono>
using namespace cv;
using namespace std;
using namespace std::chrono;
#define row 128
#define col 64
#define PI 3.14159265

#define nbin 9
#define cell_x 16
#define cell_y 16

#define block_x 2
#define block_y 2

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

void block_norm(vector<vector<double>> hog,int x,int y,vector<double>& res){
  int n=hog.size();
  vector<double> sq;
  for(int i=0;i<n;i++){
    double s=0;
    for(int j=0;j<nbin;j++)
      s+=hog[i][j]*hog[i][j];
    sq.push_back(s);
  }
  for(int i=0;i<n;i++){
    if(x-(i%x)<block_x)
      continue;
    if(i+x*(block_y-1)>=n)
      break;
    double s=0;
    for(int j=0;j<block_y;j++){
      int ind=j*x;
      for(int k=0;k<block_x;k++){
        s+=sq[i+ind+k];
      }
    }
    s=sqrt(s);
    for(int j=0;j<block_y;j++){
      int ind=j*x;
      for(int k=0;k<block_x;k++){
        for(int l=0;l<nbin;l++){
          if(s)
          res.push_back(hog[i+ind+k][l]/s);
          //cout<<hog[i+ind+k][l]/s;
          else
          //cout<<0;
          res.push_back(0);
        }
      }
    }
    //cout<<s<<"\n";
  }
}

void calc_hog(Mat img,vector<double>& res){
    vector<vector<double>> hogVec;

    cvtColor(img,img,COLOR_BGR2GRAY);

    img.convertTo(img,CV_32F,1/255.0);

    //Calculate gradients in x and y direction
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    //Calculate gradient mag. and direction
    Mat mag, ang;
    cartToPolar(gx, gy, mag, ang, 1);
    for(int i=0;i<mag.rows;i+=cell_y){
      for(int j=0;j<mag.cols;j+=cell_x){
          vector<double> bin;
          Mat cell_mag=mag.rowRange(i,i+cell_y).colRange(j,j+cell_x);
          Mat cell_ang=ang.rowRange(i,i+cell_y).colRange(j,j+cell_x);
          bin=getFeature(cell_mag,cell_ang);
          hogVec.push_back(bin);
      }
    }
    block_norm(hogVec,mag.cols/cell_x,mag.rows/cell_y,res);
}

int main(int argc,char **argv){

  Mat img;
  img=imread(argv[1],1);

  vector<double> fd;
  int64 t0 = cv::getTickCount();

  calc_hog(img,fd);
  int64 t1 = cv::getTickCount();

  double secs = (t1-t0)/cv::getTickFrequency();

  cout<< "\nProcessing time on CPU (ms): " << secs*1000 << "\n";

  cout<<fd.size()<<"\n";
  /*
  ofstream file;
  file.open("feat.dat");
  auto start = high_resolution_clock::now();
  for(int i=1;i<=400;i++){
    Mat img=imread("images/"+to_string(i)+".jpg");
    cout<<i<<"\n";
    //cout<<img.cols;
    vector<double> fd;
    calc_hog(img,fd);


    for(int i=0;i<fd.size();i++)
      file<<fd[i]<<" ";
    file<<"\n";

  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout<<duration.count()<<"\n";
  file.close();*/

  ofstream file;
  file.open("fd1.dat");
  for(int i=0;i<fd.size();i++)
    file<<fd[i]<<"\n";
  file.close();

  return 0;
}
