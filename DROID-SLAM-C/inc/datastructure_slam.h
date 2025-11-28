/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#ifndef __DATA_STRUCTURE_SLAM__
#define __DATA_STRUCTURE_SLAM__

#include <vector>
#include <cstring>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <unordered_set>



#include "bounding_box.hpp"
#include "object.hpp"
#include "syslog.h"

#define MAX_NUM_OBJ 100 // 32


// Alister add 2025-11-28
struct Tensor4D {
    int B=0, C=0, H=0, W=0;
    Tensor4D() {} // default constructor
    std::vector<float> data;  // contiguous memory

    Tensor4D(int B_, int C_, int H_, int W_) : B(B_), C(C_), H(H_), W(W_), data(B_*C_*H_*W_, 0.0f) {}

    inline float& operator()(int b,int c,int h,int w) {
        return data[((b*C + c)*H + h)*W + w];
    }
};

struct Tensor5D {
    int B=0, N=0, C=0, H=0, W=0;
    Tensor5D() {} // default constructor
    std::vector<float> data;

    Tensor5D(int B_, int N_, int C_, int H_, int W_) : B(B_), N(N_), C(C_), H(H_), W(W_), data(B_*N_*C_*H_*W_, 0.0f) {}

    inline float& operator()(int b,int n,int c,int h,int w) {
        return data[((((b*N + n)*C + c)*H + h)*W + w)];
    }
};


// For simplicity, define camera intrinsics
struct Intrinsics {
    float fx, fy, cx, cy;
};


struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& r) const { return Vec3(x+r.x, y+r.y, z+r.z); }
    Vec3 operator-(const Vec3& r) const { return Vec3(x-r.x, y-r.y, z-r.z); }
    Vec3 operator*(float s)      const { return Vec3(x*s, y*s, z*s); }
};


struct SE3 {
    // rotation 3x3
    float R[3][3];
    // translation
    Vec3 t;

    // world = R * local + t
    Vec3 toWorld(const Vec3& p) const {
        Vec3 r;
        r.x = R[0][0]*p.x + R[0][1]*p.y + R[0][2]*p.z + t.x;
        r.y = R[1][0]*p.x + R[1][1]*p.y + R[1][2]*p.z + t.y;
        r.z = R[2][0]*p.x + R[2][1]*p.y + R[2][2]*p.z + t.z;
        return r;
    }

    // local = R^T * (world - t)
    Vec3 fromWorld(const Vec3& p) const {
        Vec3 q = p - t;

        Vec3 r;
        r.x = R[0][0]*q.x + R[1][0]*q.y + R[2][0]*q.z;
        r.y = R[0][1]*q.x + R[1][1]*q.y + R[2][1]*q.z;
        r.z = R[0][2]*q.x + R[1][2]*q.y + R[2][2]*q.z;
        return r;
    }
};

struct UpdateResult {
    Tensor4D net;
    Tensor5D delta;
    Tensor4D weight;
    std::vector<float> damping;
    std::vector<int> upmask;
};

#endif
