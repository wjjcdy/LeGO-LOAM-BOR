// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar
//   Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems
//      (IROS). October 2018.

#include "featureAssociation.h"

const float RAD2DEG = 180.0 / M_PI;

FeatureAssociation::FeatureAssociation(ros::NodeHandle &node,
                                       Channel<ProjectionOut> &input_channel,
                                       Channel<AssociationOut> &output_channel)
    : nh(node),
      _input_channel(input_channel),
      _output_channel(output_channel) {

  // 全部为发出，接收的数据，以管道在input_channel接收
  pubCornerPointsSharp =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
  pubCornerPointsLessSharp =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
  pubSurfPointsFlat =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
  pubSurfPointsLessFlat =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

  _pub_cloud_corner_last =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
  _pub_cloud_surf_last =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
  _pub_outlier_cloudLast =
      nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
  pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 5);

  _cycle_count = 0;

  // 雷达相关参数
  nh.getParam("/lego_loam/laser/num_vertical_scans", _vertical_scans);
  nh.getParam("/lego_loam/laser/num_horizontal_scans", _horizontal_scans);
  nh.getParam("/lego_loam/laser/scan_period", _scan_period);

  // 一些阈值参数
  nh.getParam("/lego_loam/featureAssociation/edge_threshold", _edge_threshold);    // 边缘特征阈值0.1
  nh.getParam("/lego_loam/featureAssociation/surf_threshold", _surf_threshold);    // 表面平面阈值0.1

  nh.getParam("/lego_loam/mapping/mapping_frequency_divider", _mapping_frequency_div);

  float nearest_dist;
  nh.getParam("/lego_loam/featureAssociation/nearest_feature_search_distance", nearest_dist);
  _nearest_feature_dist_sqr = nearest_dist*nearest_dist;                           // 最近搜索距离的平方

  //变量初始化
  initializationValue();

 //开辟一个线程用于循环处理，因为无topic callback
 _run_thread = std::thread (&FeatureAssociation::runFeatureAssociation, this);
}

FeatureAssociation::~FeatureAssociation()
{
  _input_channel.send({});
  _run_thread.join();
}

void FeatureAssociation::initializationValue() {
  const size_t cloud_size = _vertical_scans * _horizontal_scans;
  cloudSmoothness.resize(cloud_size);

  // 降采样参数设置
  downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

  // 分割点云和未分割点云
  segmentedCloud.reset(new pcl::PointCloud<PointType>());
  outlierCloud.reset(new pcl::PointCloud<PointType>());

  // 角点和平面特征点点云
  cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
  cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
  surfPointsFlat.reset(new pcl::PointCloud<PointType>());
  surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

  surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
  surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

  cloudCurvature.resize(cloud_size);
  cloudNeighborPicked.resize(cloud_size);
  cloudLabel.resize(cloud_size);

  pointSelCornerInd.resize(cloud_size);
  pointSearchCornerInd1.resize(cloud_size);
  pointSearchCornerInd2.resize(cloud_size);

  pointSelSurfInd.resize(cloud_size);
  pointSearchSurfInd1.resize(cloud_size);
  pointSearchSurfInd2.resize(cloud_size);
  pointSearchSurfInd3.resize(cloud_size);

  systemInitCount = 0;
  systemInited = false;

  skipFrameNum = 1;

  // 位姿初始化，包括
  for (int i = 0; i < 6; ++i) {
    transformCur[i] = 0;    // 当前帧相对上一帧的状态转移矩阵
    transformSum[i] = 0;    // 当前帧相对第一帧的状态转移矩阵，即全局位姿
  }

  // 设置
  systemInitedLM = false;

  laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
  laserCloudOri.reset(new pcl::PointCloud<PointType>()); 
  coeffSel.reset(new pcl::PointCloud<PointType>());

  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";

  laserOdometryTrans.frame_id_ = "/camera_init";
  laserOdometryTrans.child_frame_id_ = "/laser_odom";

  isDegenerate = false;

  frameCount = skipFrameNum;
}


// 坐标系更改，不是ros中常用的右手坐标系
void FeatureAssociation::adjustDistortion() {
  bool halfPassed = false;
  int cloudSize = segmentedCloud->points.size();      //对分类

  PointType point;

  for (int i = 0; i < cloudSize; i++) {               // 坐标系做了重新调整
    point.x = segmentedCloud->points[i].y;            // 朝左为x
    point.y = segmentedCloud->points[i].z;            // 朝上为y
    point.z = segmentedCloud->points[i].x;            // 朝前为z

    float ori = -atan2(point.x, point.z);             // 表明atan2（y，x），表明绕z轴的旋转，范围为-PI～ PI
    if (!halfPassed) {                                // 表明未经过一半的角度
      if (ori < segInfo.startOrientation - M_PI / 2)
        ori += 2 * M_PI;
      else if (ori > segInfo.startOrientation + M_PI * 3 / 2)
        ori -= 2 * M_PI;

      if (ori - segInfo.startOrientation > M_PI) halfPassed = true;
    } else {
      ori += 2 * M_PI;

      if (ori < segInfo.endOrientation - M_PI * 3 / 2)
        ori += 2 * M_PI;
      else if (ori > segInfo.endOrientation + M_PI / 2)
        ori -= 2 * M_PI;
    }
    // 计算当前point在整个水平扫描位置比例，或者说扫描的时刻在一圈中的比例
    float relTime = (ori - segInfo.startOrientation) / segInfo.orientationDiff;

    // 由于扫描周期为_scan_period，故_scan_period * relTime为当前点扫描的时刻， 而原始强度为扫描索引序号的变形，由于取整数，因此原始强度仅是代表扫描的行数
    point.intensity =
        int(segmentedCloud->points[i].intensity) + _scan_period * relTime;    // 强度信息与随着每个点扫描角度不同变大

    segmentedCloud->points[i] = point;               //将坐标系进行调整保存
  }
}

// 用于计算平滑度，即当前点与前后各5个点距离差
void FeatureAssociation::calculateSmoothness() {
  int cloudSize = segmentedCloud->points.size();                         // 坐标系已经发生改变
  for (int i = 5; i < cloudSize - 5; i++) {                              // 剔除左右两侧的5个端点
    float diffRange = segInfo.segmentedCloudRange[i - 5] +               // 当前点的10倍与水平方向上前后5个点，距离差。差越大，表明当前点为1边缘点
                      segInfo.segmentedCloudRange[i - 4] +               // 连续点云边缘的点最大
                      segInfo.segmentedCloudRange[i - 3] +
                      segInfo.segmentedCloudRange[i - 2] +
                      segInfo.segmentedCloudRange[i - 1] -
                      segInfo.segmentedCloudRange[i] * 10 +
                      segInfo.segmentedCloudRange[i + 1] +
                      segInfo.segmentedCloudRange[i + 2] +
                      segInfo.segmentedCloudRange[i + 3] +
                      segInfo.segmentedCloudRange[i + 4] +
                      segInfo.segmentedCloudRange[i + 5];

    cloudCurvature[i] = diffRange * diffRange;                          // 获取当前曲率

    cloudNeighborPicked[i] = 0;                                         // 边界点标记，不稳定的点和已处理特征后的点
    cloudLabel[i] = 0;

    cloudSmoothness[i].value = cloudCurvature[i];
    cloudSmoothness[i].ind = i;
  }
}

//标记水平方向上相邻的两点差距较大时标记为1
void FeatureAssociation::markOccludedPoints() {
  int cloudSize = segmentedCloud->points.size();

  for (int i = 5; i < cloudSize - 6; ++i) {
    float depth1 = segInfo.segmentedCloudRange[i];
    float depth2 = segInfo.segmentedCloudRange[i + 1];                    // 遍历提取相邻的两个点的距离
    int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i + 1] -   // 获取相邻两个点的在水平方向上的索引号差
                                  segInfo.segmentedCloudColInd[i]));

    if (columnDiff < 10) {                                                // 如果水平索引在10个点内，将远处的边缘的5个点标记为1,
      if (depth1 - depth2 > 0.3) {
        cloudNeighborPicked[i - 5] = 1;                                   // 近处的边缘点则为0
        cloudNeighborPicked[i - 4] = 1;
        cloudNeighborPicked[i - 3] = 1;
        cloudNeighborPicked[i - 2] = 1;
        cloudNeighborPicked[i - 1] = 1;
        cloudNeighborPicked[i] = 1;
      } else if (depth2 - depth1 > 0.3) {
        cloudNeighborPicked[i + 1] = 1;
        cloudNeighborPicked[i + 2] = 1;
        cloudNeighborPicked[i + 3] = 1;
        cloudNeighborPicked[i + 4] = 1;
        cloudNeighborPicked[i + 5] = 1;
        cloudNeighborPicked[i + 6] = 1;
      }
    }

    float diff1 = std::abs(segInfo.segmentedCloudRange[i - 1] -
                           segInfo.segmentedCloudRange[i]);
    float diff2 = std::abs(segInfo.segmentedCloudRange[i + 1] -
                           segInfo.segmentedCloudRange[i]);

    if (diff1 > 0.02 * segInfo.segmentedCloudRange[i] &&                  // 如果两个相邻点的距离超过本点的距离0.02倍时，也标记为1，即孤立的点标记为1
        diff2 > 0.02 * segInfo.segmentedCloudRange[i])
      cloudNeighborPicked[i] = 1;
  }
}

// 特征提取
void FeatureAssociation::extractFeatures() {
  cornerPointsSharp->clear();         // 角点
  cornerPointsLessSharp->clear();     // 轻微角点
  surfPointsFlat->clear();            // 地面上平摊的点
  surfPointsLessFlat->clear();        // 非角点

  for (int i = 0; i < _vertical_scans; i++) {                                    // 外循环为每一束激光器
    surfPointsLessFlatScan->clear();

    for (int j = 0; j < 6; j++) {                                                // 将每一激光束等分成6组进行处理，等分原理未明白
      int sp =
          (segInfo.startRingIndex[i] * (6 - j) + segInfo.endRingIndex[i] * j) /  // 获取每一组起点坐标索引
          6;
      int ep = (segInfo.startRingIndex[i] * (5 - j) +                            // 获取每一组终点坐标索引
                segInfo.endRingIndex[i] * (j + 1)) /
                   6 -
               1;

      if (sp >= ep) continue;                                                    // 若起点索引大于终点索引，则无需处理

      std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep,      // 每组根据弧度进行从小到大排序
                by_value());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {                                           // 将弧度由大到小
        int ind = cloudSmoothness[k].ind;                                        // 提取索引
        if (cloudNeighborPicked[ind] == 0 &&                                     // 当前并非远方点变近的边缘的点、并且平滑度大于一定值、非地面数据（结论：就是提取水平方向连续断开的端点，且仅提取断开近处的端点）
            cloudCurvature[ind] > _edge_threshold &&                             // ？？？？？？？这里是提取出连续点中断开的端点，即连续点云断开的边缘点？？？？
            segInfo.segmentedCloudGroundFlag[ind] == false) {
          largestPickedNum++;                                                    // 统计满足上条件的个数
          if (largestPickedNum <= 2) {                                           // 记录最大的两个点，为最陡的两个点，即一圈最多6*2 = 12个点
            cloudLabel[ind] = 2;
            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
          } else if (largestPickedNum <= 20) {                                   // 前20个点为稍微陡峭的两个点， 
            cloudLabel[ind] = 1;
            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
          } else {                                                               // 后面无需考虑
            break;
          }

          cloudNeighborPicked[ind] = 1;                                          // 此点近距离边缘端点，已经处理过并将其设置为1
          for (int l = 1; l <= 5; l++) {                                         // 如果当前点在后5个点内，不处理
            if( ind + l >= segInfo.segmentedCloudColInd.size() ) {
              continue;
            }
            int columnDiff =
                std::abs(int(segInfo.segmentedCloudColInd[ind + l] -
                             segInfo.segmentedCloudColInd[ind + l - 1]));        // 如果相邻有效索引的两点，在水平上索超出10（即10个水平角度分辨率），便跳出，即不平坦
            if (columnDiff > 10) break;
            cloudNeighborPicked[ind + l] = 1;                                    // 即与当前点相对平滑的5个点设置为1，
          }
          for (int l = -1; l >= -5; l--) {                                       // 若当前点在前5个点内，不处理
            if( ind + l < 0 ) {
              continue;
            }
            int columnDiff =
                std::abs(int(segInfo.segmentedCloudColInd[ind + l] -             // 同理 
                             segInfo.segmentedCloudColInd[ind + l + 1]));
            if (columnDiff > 10) break;
            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      int smallestPickedNum = 0;                                                 // 经过以上处理 cloudNeighborPicked = 0， 表明均不是连续点云中边缘处的5个点
      for (int k = sp; k <= ep; k++) {                                           // 平滑度从小到大遍历
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < _surf_threshold &&                             // 平滑的点，且是地面上的点
            segInfo.segmentedCloudGroundFlag[ind] == true) {
          cloudLabel[ind] = -1;                                                  // 标记平滑地面的点为-1
          surfPointsFlat->push_back(segmentedCloud->points[ind]);                // 放入平滑地面点云

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {                                          // 即一圈最多为6*4 = 24个点
            break;
          }

          cloudNeighborPicked[ind] = 1;                                          // 表明此点已被处理
          for (int l = 1; l <= 5; l++) {
            if( ind + l >= segInfo.segmentedCloudColInd.size() ) {               // 超出边界无需关心
              continue;
            }
            int columnDiff =
                std::abs(int(segInfo.segmentedCloudColInd.at(ind + l) -
                             segInfo.segmentedCloudColInd.at(ind + l - 1)));
            if (columnDiff > 10) break;                                           // 如果相邻的两个有效点在扫描索引相差10坐标个以上的，则无需间隔

            cloudNeighborPicked[ind + l] = 1;                                     // 此操作可作为降采样功能，使特征点至少间隔5个点
          }
          for (int l = -1; l >= -5; l--) {
            if (ind + l < 0) {
              continue;
            }
            int columnDiff =
                std::abs(int(segInfo.segmentedCloudColInd.at(ind + l) -
                             segInfo.segmentedCloudColInd.at(ind + l + 1)));
            if (columnDiff > 10) break;

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {                                             // 除去陡峭的边缘点，均为，稍微平面的点
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
        }
      }
    }

    surfPointsLessFlatScanDS->clear();
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);                          // 降采样处理
    downSizeFilter.filter(*surfPointsLessFlatScanDS);

    *surfPointsLessFlat += *surfPointsLessFlatScanDS;                              // 每次降采样一行，进行拼接
  }
}

// loam假设激光雷达匀速运动，因此采用匀速模型对点云进行畸变修正

// 将当前帧的每个点相对于该扫描行的第一个点去除运动畸变
void FeatureAssociation::TransformToStart(PointType const *const pi,
                                          PointType *const po) {
  // 插值系数，为相对与第一个点的时间差
  // 其中10为系数，可调整，感觉像假设10是个速度？？？？？
  // 表明
  float s = 10 * (pi->intensity - int(pi->intensity));       // 由于强度为thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0; // 强度根据不同层则不一样，可用于显示区分
                                                             // 而在特征提取进行了修改 intensity = int(segmentedCloud->points[i].intensity) + _scan_period * relTime; 
                                                             // 如此可看出， s应该为每扫描行中点云与第一个点云的时间差即_scan_period * relTime
  // 线性插值， 如果静止，显然无需插值，则每个点相对激光原点（或者相对于上刻位置），全部一致
  // 由于匀速运动，因此认为点云中的每个点相对于上刻的位置有一定的偏移，其偏移量为s
  float ry = s * transformCur[1];
  float rx = s * transformCur[0];                            
  float rz = s * transformCur[2];
  float tx = s * transformCur[3];
  float ty = s * transformCur[4];
  float tz = s * transformCur[5];

  // 将该点根据自己的畸变进行旋转和平移
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;
  po->intensity = pi->intensity;
}

// 将上一帧中的每个点相对于该扫描行的最后一个点去除运动畸变， 如此前后两帧的则在同一个坐标系下进行畸变校正
// 输入的为上一帧的点云
void FeatureAssociation::TransformToEnd(PointType const *const pi,
                                        PointType *const po) {
  // 相对于第一个点进行校准
  float s = 10 * (pi->intensity - int(pi->intensity));

  float rx = s * transformCur[0];
  float ry = s * transformCur[1];
  float rz = s * transformCur[2];
  float tx = s * transformCur[3];
  float ty = s * transformCur[4];
  float tz = s * transformCur[5];

  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2;

  // 根据第一个点，也为上帧最后一点，进行反向转换
  rx = transformCur[0];
  ry = transformCur[1];
  rz = transformCur[2];
  tx = transformCur[3];
  ty = transformCur[4];
  tz = transformCur[5];

  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;

  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

  po->x = x6;
  po->y = y6;
  po->z = z6;
  // 取整表明仅保留扫描线的行号
  po->intensity = int(pi->intensity);
}


void FeatureAssociation::AccumulateRotation(float cx, float cy, float cz,
                                            float lx, float ly, float lz,
                                            float &ox, float &oy, float &oz) {
  float srx = cos(lx) * cos(cx) * sin(ly) * sin(cz) -
              cos(cx) * cos(cz) * sin(lx) - cos(lx) * cos(ly) * sin(cx);
  ox = -asin(srx);

  float srycrx =
      sin(lx) * (cos(cy) * sin(cz) - cos(cz) * sin(cx) * sin(cy)) +
      cos(lx) * sin(ly) * (cos(cy) * cos(cz) + sin(cx) * sin(cy) * sin(cz)) +
      cos(lx) * cos(ly) * cos(cx) * sin(cy);
  float crycrx =
      cos(lx) * cos(ly) * cos(cx) * cos(cy) -
      cos(lx) * sin(ly) * (cos(cz) * sin(cy) - cos(cy) * sin(cx) * sin(cz)) -
      sin(lx) * (sin(cy) * sin(cz) + cos(cy) * cos(cz) * sin(cx));
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

  float srzcrx =
      sin(cx) * (cos(lz) * sin(ly) - cos(ly) * sin(lx) * sin(lz)) +
      cos(cx) * sin(cz) * (cos(ly) * cos(lz) + sin(lx) * sin(ly) * sin(lz)) +
      cos(lx) * cos(cx) * cos(cz) * sin(lz);
  float crzcrx =
      cos(lx) * cos(lz) * cos(cx) * cos(cz) -
      cos(cx) * sin(cz) * (cos(ly) * sin(lz) - cos(lz) * sin(lx) * sin(ly)) -
      sin(cx) * (sin(ly) * sin(lz) + cos(ly) * cos(lz) * sin(lx));
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}

// 查找对应的直线特征，并计算目标点云中每个点到达上帧的点云的对应关系，即点到线的距离
void FeatureAssociation::findCorrespondingCornerFeatures(int iterCount) {
  int cornerPointsSharpNum = cornerPointsSharp->points.size();

  for (int i = 0; i < cornerPointsSharpNum; i++) {
    PointType pointSel;
    TransformToStart(&cornerPointsSharp->points[i], &pointSel);

    // 从上帧点云中找到与当前帧一个点最近的两个对应点
    if (iterCount % 5 == 0) {
      kdtreeCornerLast.nearestKSearch(pointSel, 1, pointSearchInd,
                                       pointSearchSqDis);
      int closestPointInd = -1, minPointInd2 = -1;

      if (pointSearchSqDis[0] < _nearest_feature_dist_sqr) {
        closestPointInd = pointSearchInd[0];                               // 最近点索引
        int closestPointScan =                                             // 最近点scan行号
            int(laserCloudCornerLast->points[closestPointInd].intensity);

        float pointSqDis, minPointSqDis2 = _nearest_feature_dist_sqr;
        for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {
          if (int(laserCloudCornerLast->points[j].intensity) >             // 查找两行以内的点云
              closestPointScan + 2.5) {
            break;
          }

          pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                           (laserCloudCornerLast->points[j].x - pointSel.x) +
                       (laserCloudCornerLast->points[j].y - pointSel.y) *
                           (laserCloudCornerLast->points[j].y - pointSel.y) +
                       (laserCloudCornerLast->points[j].z - pointSel.z) *
                           (laserCloudCornerLast->points[j].z - pointSel.z);

          if (int(laserCloudCornerLast->points[j].intensity) >
              closestPointScan) {
            if (pointSqDis < minPointSqDis2) {
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            }
          }
        }
        for (int j = closestPointInd - 1; j >= 0; j--) {
          if (int(laserCloudCornerLast->points[j].intensity) <
              closestPointScan - 2.5) {
            break;
          }

          pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                           (laserCloudCornerLast->points[j].x - pointSel.x) +
                       (laserCloudCornerLast->points[j].y - pointSel.y) *
                           (laserCloudCornerLast->points[j].y - pointSel.y) +
                       (laserCloudCornerLast->points[j].z - pointSel.z) *
                           (laserCloudCornerLast->points[j].z - pointSel.z);

          if (int(laserCloudCornerLast->points[j].intensity) <
              closestPointScan) {
            if (pointSqDis < minPointSqDis2) {
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            }
          }
        }
      }

      pointSearchCornerInd1[i] = closestPointInd;                 //最近点的索引 
      pointSearchCornerInd2[i] = minPointInd2;                    //第二近点的索引
    }

    // 如果次近点有效
    if (pointSearchCornerInd2[i] >= 0) {
      PointType tripod1 =
          laserCloudCornerLast->points[pointSearchCornerInd1[i]];
      PointType tripod2 =
          laserCloudCornerLast->points[pointSearchCornerInd2[i]];
      // 目标点M
      float x0 = pointSel.x;
      float y0 = pointSel.y;
      float z0 = pointSel.z;
      // 匹配对应的两个点 B 和 A
      float x1 = tripod1.x;      // B点
      float y1 = tripod1.y;
      float z1 = tripod1.z;
      float x2 = tripod2.x;      // A点
      float y2 = tripod2.y;
      float z2 = tripod2.z;

      // 向量AM （x0 - x2）, (y0 - y2), (z0 - z2) 
      // 向量AB （x1 - x2）, (y1 - y2), (z1 - z2)

      // 代码却是 MA × MB， 求出法向量 
      // x 分量
      float m11 = ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1));
      // y 分量
      float m22 = ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1));
      // z 分量
      float m33 = ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1));

      // 法向量的模
      float a012 = sqrt(m11 * m11 + m22 * m22 + m33 * m33);

      // 向量AB的模
      float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                       (z1 - z2) * (z1 - z2));

      // ？？？？不太理解
      // //AB方向的单位向量与OAB平面的单位法向量的向量积在各轴上的分量（d的方向）他人解释：https://zhuanlan.zhihu.com/p/145858473?from_voters_page=true
      float la = ((y1 - y2) * m11 + (z1 - z2) * m22) / a012 / l12;

      float lb = -((x1 - x2) * m11 - (z1 - z2) * m33) / a012 / l12;

      float lc = -((x1 - x2) * m22 + (y1 - y2) * m33) / a012 / l12;

      // 点到直线的距离, 公式为|AM × AB| / |AB|,其中AB为匹配的对应的两个点
      // 但是a012 却是[MA × MB]， 从一个面的法向量来看，应该是一样的 
      float ld2 = a012 / l12;

      // 计算权重
      float s = 1;
      if (iterCount >= 5) {
        s = 1 - 1.8 * fabs(ld2);
      }

      // 记录权重大的点特征点对应关系
      if (s > 0.1 && ld2 != 0) {
        PointType coeff;
        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        laserCloudOri->push_back(cornerPointsSharp->points[i]);
        coeffSel->push_back(coeff);
      }
    }
  }
}

// 查找对应的平面特征，并计算目标点云中每个点到达上帧的点云的对应关系，即点到面距离
void FeatureAssociation::findCorrespondingSurfFeatures(int iterCount) {
  int surfPointsFlatNum = surfPointsFlat->points.size();

  for (int i = 0; i < surfPointsFlatNum; i++) {                                // 遍历平面点云中每个点
    PointType pointSel;
    TransformToStart(&surfPointsFlat->points[i], &pointSel);                   // 将每个点转换为以当前视角下的坐标

    // 每5次迭代搜索上帧点云中与其对应的最近的一个点，和最近点扫描内环和外环最近的各一个点
    // 即求出对应点在上帧中最近的不同扫描环的的3个点
    if (iterCount % 5 == 0) {                                                  // 每5帧进行一次处理
      kdtreeSurfLast.nearestKSearch(pointSel, 1, pointSearchInd,               // 采用Kd树在上一帧地面点云中，找到当前平面点中最近的一点
                                     pointSearchSqDis);
      int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

      if (pointSearchSqDis[0] < _nearest_feature_dist_sqr) {                   // 最近点距离满足设置条件
        closestPointInd = pointSearchInd[0];                                   // 最近点的索引ID
        int closestPointScan =                                                 // 最近点的行ID
            int(laserCloudSurfLast->points[closestPointInd].intensity);

        float pointSqDis, minPointSqDis2 = _nearest_feature_dist_sqr,          // 赋值最近距离
                          minPointSqDis3 = _nearest_feature_dist_sqr;
        for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {        // 从最近点索引开始搜索在相邻两行内的所有平面点
          if (int(laserCloudSurfLast->points[j].intensity) >
              closestPointScan + 2.5) {
            break;
          }

          pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *        // 获取上帧中附近点到此点距离
                           (laserCloudSurfLast->points[j].x - pointSel.x) +
                       (laserCloudSurfLast->points[j].y - pointSel.y) *
                           (laserCloudSurfLast->points[j].y - pointSel.y) +
                       (laserCloudSurfLast->points[j].z - pointSel.z) *
                           (laserCloudSurfLast->points[j].z - pointSel.z);

          if (int(laserCloudSurfLast->points[j].intensity) <=                 // 如果判断的点行号小于最近点行号
              closestPointScan) {
            if (pointSqDis < minPointSqDis2) {
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;                                               // 获取最近点行以内（更靠内环的点云）中最近的点
            }
          } else {
            if (pointSqDis < minPointSqDis3) {                                // 获取最近点行以外（更靠外环的点云）中最近的点 
              minPointSqDis3 = pointSqDis;
              minPointInd3 = j;
            }
          }
        }
        for (int j = closestPointInd - 1; j >= 0; j--) {                      // 遍历比最近点内环内所有点云
          if (int(laserCloudSurfLast->points[j].intensity) <
              closestPointScan - 2.5) {
            break;
          }

          pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                           (laserCloudSurfLast->points[j].x - pointSel.x) +
                       (laserCloudSurfLast->points[j].y - pointSel.y) *
                           (laserCloudSurfLast->points[j].y - pointSel.y) +
                       (laserCloudSurfLast->points[j].z - pointSel.z) *
                           (laserCloudSurfLast->points[j].z - pointSel.z);

          if (int(laserCloudSurfLast->points[j].intensity) >=
              closestPointScan) {
            if (pointSqDis < minPointSqDis2) {
              minPointSqDis2 = pointSqDis;
              minPointInd2 = j;
            }
          } else {
            if (pointSqDis < minPointSqDis3) {
              minPointSqDis3 = pointSqDis;
              minPointInd3 = j;
            }
          }
        }
      }

      pointSearchSurfInd1[i] = closestPointInd;       // 最近点索引ID
      pointSearchSurfInd2[i] = minPointInd2;          // 内环最近点索引ID
      pointSearchSurfInd3[i] = minPointInd3;          // 外环最近点索引ID
    }

    // 如果内环和外环最近点存在有效，进行匹配处理
    if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
      // 获取三个点，构成一个平面的三角形
      PointType tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]]; // A点
      PointType tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]]; // B点
      PointType tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]]; // C点


      // 平面向量
      // AB (tripod2.x - tripod1.x, tripod2.y - tripod1.y, tripod2.z - tripod1.z)
      // AC (tripod3.x - tripod1.x, tripod3.y - tripod1.y, tripod3.z - tripod1.z)

      // 向量叉乘获得平面法向量AB × AC
      // 法向量三个分量(可套公式获得)
      float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) -
                 (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
      float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) -
                 (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
      float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) -
                 (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
            
      // 平面方程 Ax+By+Cz+D = 0      
      // 获取平面方程的常数D
      float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

      // 平面法向量的模
      float ps = sqrt(pa * pa + pb * pb + pc * pc);

      // 换算成单位向量
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      // 计算目标点到达平面的距离， 即目标点在方向量上的投影
      float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

      // 考虑权重，距离越小，权重越大，应该是经验值??????
      float s = 1;
      if (iterCount >= 5) {
        s = 1 -
            1.8 * fabs(pd2) /
                sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y +
                          pointSel.z * pointSel.z));
      }

      // 记录平面参数和点到平面的距离,权重过小则忽略
      if (s > 0.1 && pd2 != 0) {
        PointType coeff;
        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        laserCloudOri->push_back(surfPointsFlat->points[i]);        // 存放原始点云
        coeffSel->push_back(coeff);                                 // 存放平面信息和点到平面距离
      }
    }
  }
}

// 平面匹配LM优化
// AX = B , 求解采用QR分解，即最小二乘解法
// A = [J] 即偏导数
// B 为点到对应平面的距离乘以权重
bool FeatureAssociation::calculateTransformationSurf(int iterCount) {
  int pointSelNum = laserCloudOri->points.size();

  Eigen::Matrix<float,Eigen::Dynamic,3> matA(pointSelNum, 3);
  Eigen::Matrix<float,3,Eigen::Dynamic> matAt(3,pointSelNum);
  Eigen::Matrix<float,3,3> matAtA;
  Eigen::VectorXf matB(pointSelNum);
  Eigen::Matrix<float,3,1> matAtB;
  Eigen::Matrix<float,3,1> matX;
  Eigen::Matrix<float,3,3> matP;

  // 中间变量
  float srx = sin(transformCur[0]);
  float crx = cos(transformCur[0]);
  float sry = sin(transformCur[1]);
  float cry = cos(transformCur[1]);
  float srz = sin(transformCur[2]);
  float crz = cos(transformCur[2]);
  float tx = transformCur[3];
  float ty = transformCur[4];
  float tz = transformCur[5];

  float a1 = crx * sry * srz;
  float a2 = crx * crz * sry;
  float a3 = srx * sry;
  float a4 = tx * a1 - ty * a2 - tz * a3;
  float a5 = srx * srz;
  float a6 = crz * srx;
  float a7 = ty * a6 - tz * crx - tx * a5;
  float a8 = crx * cry * srz;
  float a9 = crx * cry * crz;
  float a10 = cry * srx;
  float a11 = tz * a10 + ty * a9 - tx * a8;

  float b1 = -crz * sry - cry * srx * srz;
  float b2 = cry * crz * srx - sry * srz;
  float b5 = cry * crz - srx * sry * srz;
  float b6 = cry * srz + crz * srx * sry;

  float c1 = -b6;
  float c2 = b5;
  float c3 = tx * b6 - ty * b5;
  float c4 = -crx * crz;
  float c5 = crx * srz;
  float c6 = ty * c5 + tx * -c4;
  float c7 = b2;
  float c8 = -b1;
  float c9 = tx * -b2 - ty * -b1;

  for (int i = 0; i < pointSelNum; i++) {
    PointType pointOri = laserCloudOri->points[i];
    PointType coeff = coeffSel->points[i];
    // 偏导数
    float arx =
        (-a1 * pointOri.x + a2 * pointOri.y + a3 * pointOri.z + a4) * coeff.x +
        (a5 * pointOri.x - a6 * pointOri.y + crx * pointOri.z + a7) * coeff.y +
        (a8 * pointOri.x - a9 * pointOri.y - a10 * pointOri.z + a11) * coeff.z;

    float arz = (c1 * pointOri.x + c2 * pointOri.y + c3) * coeff.x +
                (c4 * pointOri.x - c5 * pointOri.y + c6) * coeff.y +
                (c7 * pointOri.x + c8 * pointOri.y + c9) * coeff.z;

    float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

    // 距离
    float d2 = coeff.intensity;
    // A 矩阵
    matA(i, 0) = arx;
    matA(i, 1) = arz;
    matA(i, 2) = aty;
    // B 矩阵
    matB(i, 0) = -0.05 * d2;
  }

  // A的转置
  matAt = matA.transpose();
  // A和B 同左乘A的转置， 目的是希望左侧矩阵满秩
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  // matAtA * X = matAt * matB
  // QR分解求解
  matX = matAtA.colPivHouseholderQr().solve(matAtB);

  // ????不是太明白，矩阵退化判断
  if (iterCount == 0) {
    Eigen::Matrix<float,1,3> matE;
    Eigen::Matrix<float,3,3> matV;
    Eigen::Matrix<float,3,3> matV2;
    
    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float,3,3> > esolver(matAtA);
    matE = esolver.eigenvalues().real();
    matV = esolver.eigenvectors().real();
    matV2 = matV;

    isDegenerate = false;
    float eignThre[3] = {10, 10, 10};
    for (int i = 2; i >= 0; i--) {
      if (matE(0, i) < eignThre[i]) {
        for (int j = 0; j < 3; j++) {
          matV2(i, j) = 0;
        }
        isDegenerate = true;
      } else {
        break;
      }
    }
    matP = matV.inverse() * matV2;
  }
  //
  if (isDegenerate) {
    Eigen::Matrix<float,3,1> matX2;
    matX2 = matX;
    matX = matP * matX2;
  }

  // 叠加每次迭代的旋转和平移量
  transformCur[0] += matX(0, 0);
  transformCur[2] += matX(1, 0);
  // y 轴平移 （原点云的z的平移）
  transformCur[4] += matX(2, 0);

  // 无效值就恢复0
  for (int i = 0; i < 6; i++) {
    if (std::isnan(transformCur[i])) transformCur[i] = 0;
  }

  // 计算旋转平移误差矩阵
  float deltaR = sqrt(pow(RAD2DEG * (matX(0, 0)), 2) +
                      pow(RAD2DEG * (matX(1, 0)), 2));
  float deltaT = sqrt(pow(matX(2, 0) * 100, 2));

  // 旋转平移误差小于一定阈值，则停止迭代
  if (deltaR < 0.1 && deltaT < 0.1) {
    return false;
  }
  return true;
}

// 匹配LM优化
// AX = B , 求解采用QR分解，即最小二乘解法
// A = [J] 即偏导数
// B 为点到对应线的距离乘以权重
// 与平面的LM优化完全一致
bool FeatureAssociation::calculateTransformationCorner(int iterCount) {
  int pointSelNum = laserCloudOri->points.size();

  Eigen::Matrix<float,Eigen::Dynamic,3> matA(pointSelNum, 3);
  Eigen::Matrix<float,3,Eigen::Dynamic> matAt(3,pointSelNum);
  Eigen::Matrix<float,3,3> matAtA;
  Eigen::VectorXf matB(pointSelNum);
  Eigen::Matrix<float,3,1> matAtB;
  Eigen::Matrix<float,3,1> matX;
  Eigen::Matrix<float,3,3> matP;

  float srx = sin(transformCur[0]);
  float crx = cos(transformCur[0]);
  float sry = sin(transformCur[1]);
  float cry = cos(transformCur[1]);
  float srz = sin(transformCur[2]);
  float crz = cos(transformCur[2]);
  float tx = transformCur[3];
  float ty = transformCur[4];
  float tz = transformCur[5];

  float b1 = -crz * sry - cry * srx * srz;
  float b2 = cry * crz * srx - sry * srz;
  float b3 = crx * cry;
  float b4 = tx * -b1 + ty * -b2 + tz * b3;
  float b5 = cry * crz - srx * sry * srz;
  float b6 = cry * srz + crz * srx * sry;
  float b7 = crx * sry;
  float b8 = tz * b7 - ty * b6 - tx * b5;

  float c5 = crx * srz;

  for (int i = 0; i < pointSelNum; i++) {
    PointType pointOri = laserCloudOri->points[i];
    PointType coeff = coeffSel->points[i];

    float ary =
        (b1 * pointOri.x + b2 * pointOri.y - b3 * pointOri.z + b4) * coeff.x +
        (b5 * pointOri.x + b6 * pointOri.y - b7 * pointOri.z + b8) * coeff.z;

    float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

    float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

    float d2 = coeff.intensity;

    matA(i, 0) = ary;
    matA(i, 1) = atx;
    matA(i, 2) = atz;
    matB(i, 0) = -0.05 * d2;
  }

  matAt = matA.transpose();
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  matX = matAtA.colPivHouseholderQr().solve(matAtB);

  if (iterCount == 0) {
    Eigen::Matrix<float,1, 3> matE;
    Eigen::Matrix<float,3, 3> matV;
    Eigen::Matrix<float,3, 3> matV2;

    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float,3,3> > esolver(matAtA);
    matE = esolver.eigenvalues().real();
    matV = esolver.eigenvectors().real();
    matV2 = matV;

    isDegenerate = false;
    float eignThre[3] = {10, 10, 10};
    for (int i = 2; i >= 0; i--) {
      if (matE(0, i) < eignThre[i]) {
        for (int j = 0; j < 3; j++) {
          matV2(i, j) = 0;
        }
        isDegenerate = true;
      } else {
        break;
      }
    }
    matP = matV.inverse() * matV2;
  }

  if (isDegenerate) {
    Eigen::Matrix<float,3,1> matX2;
    matX2 = matX;
    matX = matP * matX2;
  }

  transformCur[1] += matX(0, 0);
  transformCur[3] += matX(1, 0);
  transformCur[5] += matX(2, 0);

  for (int i = 0; i < 6; i++) {
    if (std::isnan(transformCur[i])) transformCur[i] = 0;
  }

  float deltaR = sqrt(pow(RAD2DEG * (matX(0, 0)), 2));
  float deltaT = sqrt(pow(matX(1, 0) * 100, 2) +
                      pow(matX(2, 0) * 100, 2));

  if (deltaR < 0.1 && deltaT < 0.1) {
    return false;
  }
  return true;
}

bool FeatureAssociation::calculateTransformation(int iterCount) {
  int pointSelNum = laserCloudOri->points.size();

  Eigen::Matrix<float,Eigen::Dynamic,6> matA(pointSelNum, 6);
  Eigen::Matrix<float,6,Eigen::Dynamic> matAt(6,pointSelNum);
  Eigen::Matrix<float,6,6> matAtA;
  Eigen::VectorXf matB(pointSelNum);
  Eigen::Matrix<float,6,1> matAtB;
  Eigen::Matrix<float,6,1> matX;
  Eigen::Matrix<float,6,6> matP;

  float srx = sin(transformCur[0]);
  float crx = cos(transformCur[0]);
  float sry = sin(transformCur[1]);
  float cry = cos(transformCur[1]);
  float srz = sin(transformCur[2]);
  float crz = cos(transformCur[2]);
  float tx = transformCur[3];
  float ty = transformCur[4];
  float tz = transformCur[5];

  float a1 = crx * sry * srz;
  float a2 = crx * crz * sry;
  float a3 = srx * sry;
  float a4 = tx * a1 - ty * a2 - tz * a3;
  float a5 = srx * srz;
  float a6 = crz * srx;
  float a7 = ty * a6 - tz * crx - tx * a5;
  float a8 = crx * cry * srz;
  float a9 = crx * cry * crz;
  float a10 = cry * srx;
  float a11 = tz * a10 + ty * a9 - tx * a8;

  float b1 = -crz * sry - cry * srx * srz;
  float b2 = cry * crz * srx - sry * srz;
  float b3 = crx * cry;
  float b4 = tx * -b1 + ty * -b2 + tz * b3;
  float b5 = cry * crz - srx * sry * srz;
  float b6 = cry * srz + crz * srx * sry;
  float b7 = crx * sry;
  float b8 = tz * b7 - ty * b6 - tx * b5;

  float c1 = -b6;
  float c2 = b5;
  float c3 = tx * b6 - ty * b5;
  float c4 = -crx * crz;
  float c5 = crx * srz;
  float c6 = ty * c5 + tx * -c4;
  float c7 = b2;
  float c8 = -b1;
  float c9 = tx * -b2 - ty * -b1;

  for (int i = 0; i < pointSelNum; i++) {
    PointType pointOri = laserCloudOri->points[i];
    PointType coeff = coeffSel->points[i];

    float arx =
        (-a1 * pointOri.x + a2 * pointOri.y + a3 * pointOri.z + a4) * coeff.x +
        (a5 * pointOri.x - a6 * pointOri.y + crx * pointOri.z + a7) * coeff.y +
        (a8 * pointOri.x - a9 * pointOri.y - a10 * pointOri.z + a11) * coeff.z;

    float ary =
        (b1 * pointOri.x + b2 * pointOri.y - b3 * pointOri.z + b4) * coeff.x +
        (b5 * pointOri.x + b6 * pointOri.y - b7 * pointOri.z + b8) * coeff.z;

    float arz = (c1 * pointOri.x + c2 * pointOri.y + c3) * coeff.x +
                (c4 * pointOri.x - c5 * pointOri.y + c6) * coeff.y +
                (c7 * pointOri.x + c8 * pointOri.y + c9) * coeff.z;

    float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

    float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

    float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

    float d2 = coeff.intensity;

    matA(i, 0) = arx;
    matA(i, 1) = ary;
    matA(i, 2) = arz;
    matA(i, 3) = atx;
    matA(i, 4) = aty;
    matA(i, 5) = atz;
    matB(i, 0) = -0.05 * d2;
  }

  matAt = matA.transpose();
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  matX = matAtA.colPivHouseholderQr().solve(matAtB);

  if (iterCount == 0) {
    Eigen::Matrix<float,1, 6> matE;
    Eigen::Matrix<float,6, 6> matV;
    Eigen::Matrix<float,6, 6> matV2;

    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float,6,6> > esolver(matAtA);
    matE = esolver.eigenvalues().real();
    matV = esolver.eigenvectors().real();
    matV2 = matV;

    isDegenerate = false;
    float eignThre[6] = {10, 10, 10, 10, 10, 10};
    for (int i = 5; i >= 0; i--) {
      if (matE(0, i) < eignThre[i]) {
        for (int j = 0; j < 6; j++) {
          matV2(i, j) = 0;
        }
        isDegenerate = true;
      } else {
        break;
      }
    }
    matP = matV.inverse() * matV2;
  }

  if (isDegenerate) {
    Eigen::Matrix<float,6,1> matX2;
    matX2 = matX;
    matX = matP * matX2;
  }

  transformCur[0] += matX(0, 0);
  transformCur[1] += matX(1, 0);
  transformCur[2] += matX(2, 0);
  transformCur[3] += matX(3, 0);
  transformCur[4] += matX(4, 0);
  transformCur[5] += matX(5, 0);

  for (int i = 0; i < 6; i++) {
    if (std::isnan(transformCur[i])) transformCur[i] = 0;
  }

  float deltaR = sqrt(pow(RAD2DEG * (matX(0, 0)), 2) +
                      pow(RAD2DEG * (matX(1, 0)), 2) +
                      pow(RAD2DEG * (matX(2, 0)), 2));
  float deltaT = sqrt(pow(matX(3, 0) * 100, 2) +
                      pow(matX(4, 0) * 100, 2) +
                      pow(matX(5, 0) * 100, 2));

  if (deltaR < 0.1 && deltaT < 0.1) {
    return false;
  }
  return true;
}

//首次初始化，即第一帧点云用于初始化，不做匹配
void FeatureAssociation::checkSystemInitialization() {
  pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;   // 第一次last 和 curr 应一样， 即赋初始状态
  cornerPointsLessSharp = laserCloudCornerLast;
  laserCloudCornerLast = laserCloudTemp;

  laserCloudTemp = surfPointsLessFlat;
  surfPointsLessFlat = laserCloudSurfLast;
  laserCloudSurfLast = laserCloudTemp;

  kdtreeCornerLast.setInputCloud(laserCloudCornerLast);                      // 初始化KD TREE
  kdtreeSurfLast.setInputCloud(laserCloudSurfLast);

  laserCloudCornerLastNum = laserCloudCornerLast->points.size();             // 获取两种特征点云个数
  laserCloudSurfLastNum = laserCloudSurfLast->points.size();

  sensor_msgs::PointCloud2 laserCloudCornerLast2;                            // 发布上次的点云集合
  pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
  laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
  laserCloudCornerLast2.header.frame_id = "/camera";
  _pub_cloud_corner_last.publish(laserCloudCornerLast2);

  sensor_msgs::PointCloud2 laserCloudSurfLast2;
  pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
  laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
  laserCloudSurfLast2.header.frame_id = "/camera";
  _pub_cloud_surf_last.publish(laserCloudSurfLast2);

  systemInitedLM = true;                                                      // 特征匹配初始化完成
}

// 更新转换矩阵，采用两次LM优化，即一次地平面和一次
// 获取当前点云与上一帧点云的转换矩阵
void FeatureAssociation::updateTransformation() {
  if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100) return;    // 点数过少不处理

  for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {                   // 迭代25次，采用第一次LM优化， 采用平面计算出 tz，tx_theta,ty_theta
    laserCloudOri->clear();                                                   // 每次都独立复位
    coeffSel->clear();

    findCorrespondingSurfFeatures(iterCount1);                                // 寻找对应平面特征

    if (laserCloudOri->points.size() < 10) continue;                          // 有效点云个数不够
    if (calculateTransformationSurf(iterCount1) == false) break;              // 计算平面之间的转换矩阵， 
  }

  for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {                   //迭代25ci 寻找对应的角点特征, 采用第二次LM计算出 tx，ty， t_yaw
    laserCloudOri->clear();
    coeffSel->clear();

    findCorrespondingCornerFeatures(iterCount2);

    if (laserCloudOri->points.size() < 10) continue;
    if (calculateTransformationCorner(iterCount2) == false) break;            // 根据各个角点特征计算转换矩阵，采用第二次LM优化
  }
}

// 矩阵转换，根据变换的矩阵， 更新全局位姿
// 先旋转， 在平移
// 已知初始全局位置和相临两帧的位姿转移矩阵，
// 累加获取当前帧全局位置
void FeatureAssociation::integrateTransformation() {
  float rx, ry, rz, tx, ty, tz;
  // 旋转累加
  AccumulateRotation(transformSum[0], transformSum[1], transformSum[2],
                     -transformCur[0], -transformCur[1], -transformCur[2], rx,
                     ry, rz);

  float x1 = cos(rz) * (transformCur[3] ) -
             sin(rz) * (transformCur[4] );
  float y1 = sin(rz) * (transformCur[3] ) +
             cos(rz) * (transformCur[4] );
  float z1 = transformCur[5];

  float x2 = x1;
  float y2 = cos(rx) * y1 - sin(rx) * z1;
  float z2 = sin(rx) * y1 + cos(rx) * z1;

  // 平移累加
  tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);   
  ty = transformSum[4] - y2;
  tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

  // 获取当前帧全局位置，即构建里程计
  transformSum[0] = rx;
  transformSum[1] = ry;     // 
  transformSum[2] = rz;     // 绕z轴方向的yaw 航向角
  transformSum[3] = tx;     // x 坐标
  transformSum[4] = ty;     // y 坐标
  transformSum[5] = tz;     // z 坐标
}

//坐标系还原成ros的右手坐标系
void FeatureAssociation::adjustOutlierCloud() {
  PointType point;
  int cloudSize = outlierCloud->points.size();
  for (int i = 0; i < cloudSize; ++i) {
    point.x = outlierCloud->points[i].y;
    point.y = outlierCloud->points[i].z;
    point.z = outlierCloud->points[i].x;
    point.intensity = outlierCloud->points[i].intensity;
    outlierCloud->points[i] = point;
  }
}

// ros 格式里程计发布
void FeatureAssociation::publishOdometry() {
  geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(    // 旋转量转换为4元数
      transformSum[2], -transformSum[0], -transformSum[1]);

  laserOdometry.header.stamp = cloudHeader.stamp;
  laserOdometry.pose.pose.orientation.x = -geoQuat.y;
  laserOdometry.pose.pose.orientation.y = -geoQuat.z;
  laserOdometry.pose.pose.orientation.z = geoQuat.x;
  laserOdometry.pose.pose.orientation.w = geoQuat.w;
  laserOdometry.pose.pose.position.x = transformSum[3];
  laserOdometry.pose.pose.position.y = transformSum[4];
  laserOdometry.pose.pose.position.z = transformSum[5];
  pubLaserOdometry.publish(laserOdometry);

  laserOdometryTrans.stamp_ = cloudHeader.stamp;                                  // 全局里程计tf变换
  laserOdometryTrans.setRotation(
      tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
  laserOdometryTrans.setOrigin(
      tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
  tfBroadcaster.sendTransform(laserOdometryTrans);
}

// ros格式发布特征点
void FeatureAssociation::publishCloud() {
  sensor_msgs::PointCloud2 laserCloudOutMsg;

  auto Publish = [&](ros::Publisher &pub,
                     const pcl::PointCloud<PointType>::Ptr &cloud) {
    if (pub.getNumSubscribers() != 0) {
      pcl::toROSMsg(*cloud, laserCloudOutMsg);
      laserCloudOutMsg.header.stamp = cloudHeader.stamp;
      laserCloudOutMsg.header.frame_id = "/camera";
      pub.publish(laserCloudOutMsg);
    }
  };

  Publish(pubCornerPointsSharp, cornerPointsSharp);             // 非常明显的角点
  Publish(pubCornerPointsLessSharp, cornerPointsLessSharp);     // 角点
  Publish(pubSurfPointsFlat, surfPointsFlat);                   // 仅地面上平面的点，实际上表示表面非常平的特征的点，
  Publish(pubSurfPointsLessFlat, surfPointsLessFlat);           // 包含地面上的平面的点
}

// 每次匹配的特征点云发布
void FeatureAssociation::publishCloudsLast() {
  // 还原运动畸变矫正点云
  int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
  for (int i = 0; i < cornerPointsLessSharpNum; i++) {
    TransformToEnd(&cornerPointsLessSharp->points[i],
                   &cornerPointsLessSharp->points[i]);
  }

  int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
  for (int i = 0; i < surfPointsLessFlatNum; i++) {
    TransformToEnd(&surfPointsLessFlat->points[i],
                   &surfPointsLessFlat->points[i]);
  }

  //记录缓存当前帧，用于下次
  pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;      // 记录此次特征，用于下次匹配
  cornerPointsLessSharp = laserCloudCornerLast;
  laserCloudCornerLast = laserCloudTemp;

  laserCloudTemp = surfPointsLessFlat;
  surfPointsLessFlat = laserCloudSurfLast;
  laserCloudSurfLast = laserCloudTemp;

  laserCloudCornerLastNum = laserCloudCornerLast->points.size();
  laserCloudSurfLastNum = laserCloudSurfLast->points.size();

  // 特征点足够多时可更新KDtree
  if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {           // 特征点足够多则可进行kdtree
    kdtreeCornerLast.setInputCloud(laserCloudCornerLast);
    kdtreeSurfLast.setInputCloud(laserCloudSurfLast);
  }

  // 还原成原ros坐标系
  frameCount++;
  adjustOutlierCloud();

  if (frameCount >= skipFrameNum + 1) {
    frameCount = 0;
    sensor_msgs::PointCloud2 cloudTemp;

    auto Publish = [&](ros::Publisher &pub,
                       const pcl::PointCloud<PointType>::Ptr &cloud) {
      if (pub.getNumSubscribers() != 0) {
        pcl::toROSMsg(*cloud, cloudTemp);
        cloudTemp.header.stamp = cloudHeader.stamp;
        cloudTemp.header.frame_id = "/camera";
        pub.publish(cloudTemp);
      }
    };

    Publish(_pub_outlier_cloudLast, outlierCloud);
    Publish(_pub_cloud_corner_last, laserCloudCornerLast);
    Publish(_pub_cloud_surf_last, laserCloudSurfLast);
  }
}

// 特征拟合主线程
void FeatureAssociation::runFeatureAssociation() {
  while (ros::ok()) {
    ProjectionOut projection;
    _input_channel.receive(projection);              // 接收三类数据

    if( !ros::ok() ) break;

    //--------------                        
    outlierCloud = projection.outlier_cloud;         //分别为未被分类的孤立点云簇
    segmentedCloud = projection.segmented_cloud;     //被分类的包含地面的点云
    segInfo = std::move(projection.seg_msg);         //分类的相关信息
    _scan_msg = std::move(projection.scan_msg);

    cloudHeader = segInfo.header;
    timeScanCur = cloudHeader.stamp.toSec();

    /**  1. Feature Extraction  */
    adjustDistortion();                              // 坐标系转换

    calculateSmoothness();                           // 计算平滑性

    markOccludedPoints();                            // 标记在水平扫描方向上，距离变化大时，将远处的5个点进行标记

    extractFeatures();                               // 特征提取包括，角点和平坦点

    publishCloud();  // cloud for visualization

    // Feature Association
    if (!systemInitedLM) {                           // 仅执行一次，用于初始化
      checkSystemInitialization();
      continue;
    }

    updateTransformation();                          // 计算两帧激光数据之间转换矩阵

    integrateTransformation();                       // 迭代更新，将每两帧之间变换，进行累计变换，构建里程计

    publishOdometry();

    publishCloudsLast();                             // cloud to mapOptimization

    // 以上为高频率的激光里程计获取
    //--------------
    _cycle_count++;

    // 建图_mapping_frequency_div倍降频更新
    if (_cycle_count == _mapping_frequency_div) {
      _cycle_count = 0;
      AssociationOut out;
      out.cloud_corner_last.reset(new pcl::PointCloud<PointType>());
      out.cloud_surf_last.reset(new pcl::PointCloud<PointType>());
      out.cloud_outlier_last.reset(new pcl::PointCloud<PointType>());

      *out.cloud_corner_last = *laserCloudCornerLast;    // 将平面信息 和角点发出给建图算法
      *out.cloud_surf_last = *laserCloudSurfLast;
      *out.cloud_outlier_last = *outlierCloud;           // 将非特征点发给建图算法

      out.laser_odometry = laserOdometry;                // 激光里程计
      //added by jiajia
      out.scan_msg.reset(new pcl::PointCloud<PointType>());
      *out.scan_msg = *_scan_msg;
      _output_channel.send(std::move(out));
    }
  }
}
