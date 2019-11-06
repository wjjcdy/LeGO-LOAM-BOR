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
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar
//   Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems
//      (IROS). October 2018.

#include "mapOptimization.h"
#include <future>

using namespace gtsam;

MapOptimization::MapOptimization(ros::NodeHandle &node,
                                 Channel<AssociationOut> &input_channel)
    : nh(node),
      _input_channel(input_channel),
      _publish_global_signal(false),
      _loop_closure_signal(false)
{
  // gtsam 
  ISAM2Params parameters;                     //在定义ISAM2实例的时候存储参数的。
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);
  
  // 关键点
  pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
  // 3d点云图，3dmap 
  pubLaserCloudSurround =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
  // 里程计
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);

  map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("/2d_map", 1, true);
  pubHistoryKeyFrames =
      nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
  pubIcpKeyFrames =
      nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
  pubRecentKeyFrames =
      nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

  downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
  downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

  // for histor key frames of loop closure
  downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
  // for surrounding key poses of scan-to-map optimization
  downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

  // 仅用于发布可视点云图的分辨率
  // for global map visualization
  downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);    // 轨迹点
  // for global map visualization 
  downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);   // 点云图

  odomAftMapped.header.frame_id = "/camera_init";
  odomAftMapped.child_frame_id = "/aft_mapped";

  aftMappedTrans.frame_id_ = "/camera_init";
  aftMappedTrans.child_frame_id_ = "/aft_mapped";

  nh.getParam("/lego_loam/laser/scan_period", _scan_period);

  nh.getParam("/lego_loam/mapping/enable_loop_closure", _loop_closure_enabled);

  nh.getParam("/lego_loam/mapping/history_keyframe_search_radius",
              _history_keyframe_search_radius);

  nh.getParam("/lego_loam/mapping/history_keyframe_search_num",
              _history_keyframe_search_num);

  nh.getParam("/lego_loam/mapping/history_keyframe_fitness_score",
              _history_keyframe_fitness_score);

  nh.getParam("/lego_loam/mapping/surrounding_keyframe_search_radius",
              _surrounding_keyframe_search_radius);                           // 不闭环，搜索当前位置附近范围内用于构建submap的点云

  nh.getParam("/lego_loam/mapping/surrounding_keyframe_search_num",           // 闭环时，存储用于submap地图的存储的最新帧点云的最大个数
              _surrounding_keyframe_search_num);

  nh.getParam("/lego_loam/mapping/global_map_visualization_search_radius",
              _global_map_visualization_search_radius);                       // 可以看到地图的最大距离，即地图边界与当前位姿的距离超出次距离不在显示

  // 内存初始化
  allocateMemory();
  
  // 地图发布进程
  _publish_global_thread = std::thread(&MapOptimization::publishGlobalMapThread, this);
  // 闭环处理进程
  _loop_closure_thread = std::thread(&MapOptimization::loopClosureThread, this);
  // slam主进程
  _run_thread = std::thread(&MapOptimization::run, this);

}

MapOptimization::~MapOptimization()
{
  _input_channel.send({});
  _run_thread.join();

  _publish_global_signal.send(false);
  _publish_global_thread.join();

  _loop_closure_signal.send(false);
  _loop_closure_thread.join();
}

// 参数初始化
void MapOptimization::allocateMemory() {
  cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
  cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

  surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
  surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

  laserCloudCornerLast.reset(
      new pcl::PointCloud<PointType>());  // corner feature set from
                                          // odoOptimization
  laserCloudSurfLast.reset(
      new pcl::PointCloud<PointType>());  // surf feature set from
                                          // odoOptimization
  laserCloudCornerLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled corner featuer set
                                          // from odoOptimization
  laserCloudSurfLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled surf featuer set from
                                          // odoOptimization
  laserCloudOutlierLast.reset(
      new pcl::PointCloud<PointType>());  // corner feature set from
                                          // odoOptimization
  laserCloudOutlierLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled corner feature set
                                          // from odoOptimization
  laserCloudSurfTotalLast.reset(
      new pcl::PointCloud<PointType>());  // surf feature set from
                                          // odoOptimization
  laserCloudSurfTotalLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled surf featuer set from
                                          // odoOptimization
  
  // added by jiajia
  _scan_msg.reset(
      new pcl::PointCloud<PointType>());
  _scan_msgDS.reset(
      new pcl::PointCloud<PointType>());

  laserCloudOri.reset(new pcl::PointCloud<PointType>());
  coeffSel.reset(new pcl::PointCloud<PointType>());

  laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());

  laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

  nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
  nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

  latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

  globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
  globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
  globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
  globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

  timeLaserOdometry = 0;
  timeLastGloalMapPublish = 0;
  timeLastProcessing = -1;

  for (int i = 0; i < 6; ++i) {
    transformLast[i] = 0;
    transformSum[i] = 0;
    transformIncre[i] = 0;
    transformTobeMapped[i] = 0;
    transformBefMapped[i] = 0;
    transformAftMapped[i] = 0;
  }


  matA0.setZero();
  matB0.fill(-1);
  matX0.setZero();

  matA1.setZero();
  matD1.setZero();
  matV1.setZero();

  isDegenerate = false;
  matP.setZero();

  laserCloudCornerFromMapDSNum = 0;
  laserCloudSurfFromMapDSNum = 0;
  laserCloudCornerLastDSNum = 0;
  laserCloudSurfLastDSNum = 0;
  laserCloudOutlierLastDSNum = 0;
  laserCloudSurfTotalLastDSNum = 0;

  potentialLoopFlag = false;
  aLoopIsClosed = false;

  latestFrameID = 0;
}

// 地图发布进程
void MapOptimization::publishGlobalMapThread()
{
  while(ros::ok())
  {
    bool ready;
    _publish_global_signal.receive(ready);    // 收到需要发送的信号
    if(ready){
      publishGlobalMap();
    }
  }
}

void MapOptimization::loopClosureThread()     // 收到闭环检测信号进行处理
{
  while(ros::ok())
  {
    bool ready;
    _loop_closure_signal.receive(ready);
    if(ready && _loop_closure_enabled){       // 如闭环使能打开，则进行闭环检测 
      performLoopClosure();
    }
  }
}

//将里程计坐标转换为地图（世界）坐标系下坐标？？？？？
void MapOptimization::transformAssociateToMap() {
  float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) -
             sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
  float y1 = transformBefMapped[4] - transformSum[4];
  float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) +
             cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

  float x2 = x1;
  float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
  float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

  transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
  transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
  transformIncre[5] = z2;

  float sbcx = sin(transformSum[0]);
  float cbcx = cos(transformSum[0]);
  float sbcy = sin(transformSum[1]);
  float cbcy = cos(transformSum[1]);
  float sbcz = sin(transformSum[2]);
  float cbcz = cos(transformSum[2]);

  float sblx = sin(transformBefMapped[0]);
  float cblx = cos(transformBefMapped[0]);
  float sbly = sin(transformBefMapped[1]);
  float cbly = cos(transformBefMapped[1]);
  float sblz = sin(transformBefMapped[2]);
  float cblz = cos(transformBefMapped[2]);

  float salx = sin(transformAftMapped[0]);
  float calx = cos(transformAftMapped[0]);
  float saly = sin(transformAftMapped[1]);
  float caly = cos(transformAftMapped[1]);
  float salz = sin(transformAftMapped[2]);
  float calz = cos(transformAftMapped[2]);

  float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz +
                       calx * calz * cblx * cblz) -
              cbcx * sbcy *
                  (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                   calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                   cblx * salx * sbly) -
              cbcx * cbcy *
                  (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                   calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                   cblx * cbly * salx);
  transformTobeMapped[0] = -asin(srx);

  float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly) -
                         cblx * sblz * (caly * calz + salx * saly * salz) +
                         calx * saly * sblx) -
                 cbcx * cbcy *
                     ((caly * calz + salx * saly * salz) *
                          (cblz * sbly - cbly * sblx * sblz) +
                      (caly * salz - calz * salx * saly) *
                          (sbly * sblz + cbly * cblz * sblx) -
                      calx * cblx * cbly * saly) +
                 cbcx * sbcy *
                     ((caly * calz + salx * saly * salz) *
                          (cbly * cblz + sblx * sbly * sblz) +
                      (caly * salz - calz * salx * saly) *
                          (cbly * sblz - cblz * sblx * sbly) +
                      calx * cblx * saly * sbly);
  float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz) -
                         cblx * cblz * (saly * salz + caly * calz * salx) +
                         calx * caly * sblx) +
                 cbcx * cbcy *
                     ((saly * salz + caly * calz * salx) *
                          (sbly * sblz + cbly * cblz * sblx) +
                      (calz * saly - caly * salx * salz) *
                          (cblz * sbly - cbly * sblx * sblz) +
                      calx * caly * cblx * cbly) -
                 cbcx * sbcy *
                     ((saly * salz + caly * calz * salx) *
                          (cbly * sblz - cblz * sblx * sbly) +
                      (calz * saly - caly * salx * salz) *
                          (cbly * cblz + sblx * sbly * sblz) -
                      calx * caly * cblx * sbly);
  transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                 crycrx / cos(transformTobeMapped[0]));

  float srzcrx =
      (cbcz * sbcy - cbcy * sbcx * sbcz) *
          (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
           calx * calz * (sbly * sblz + cbly * cblz * sblx) +
           cblx * cbly * salx) -
      (cbcy * cbcz + sbcx * sbcy * sbcz) *
          (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
           calx * salz * (cbly * cblz + sblx * sbly * sblz) +
           cblx * salx * sbly) +
      cbcx * sbcz *
          (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
  float crzcrx =
      (cbcy * sbcz - cbcz * sbcx * sbcy) *
          (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
           calx * salz * (cbly * cblz + sblx * sbly * sblz) +
           cblx * salx * sbly) -
      (sbcy * sbcz + cbcy * cbcz * sbcx) *
          (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
           calx * calz * (sbly * sblz + cbly * cblz * sblx) +
           cblx * cbly * salx) +
      cbcx * cbcz *
          (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
  transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                                 crzcrx / cos(transformTobeMapped[0]));

  x1 = cos(transformTobeMapped[2]) * transformIncre[3] -
       sin(transformTobeMapped[2]) * transformIncre[4];
  y1 = sin(transformTobeMapped[2]) * transformIncre[3] +
       cos(transformTobeMapped[2]) * transformIncre[4];
  z1 = transformIncre[5];

  x2 = x1;
  y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  transformTobeMapped[3] =
      transformAftMapped[3] -
      (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
  transformTobeMapped[4] = transformAftMapped[4] - y2;
  transformTobeMapped[5] =
      transformAftMapped[5] -
      (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
}

void MapOptimization::transformUpdate() {

  for (int i = 0; i < 6; i++) {
    transformBefMapped[i] = transformSum[i];                    // 记录里程计，即作为上次里程计坐标
    transformAftMapped[i] = transformTobeMapped[i];
  }
}

void MapOptimization::updatePointAssociateToMapSinCos() {
  cRoll = cos(transformTobeMapped[0]);
  sRoll = sin(transformTobeMapped[0]);

  cPitch = cos(transformTobeMapped[1]);
  sPitch = sin(transformTobeMapped[1]);

  cYaw = cos(transformTobeMapped[2]);
  sYaw = sin(transformTobeMapped[2]);

  tX = transformTobeMapped[3];
  tY = transformTobeMapped[4];
  tZ = transformTobeMapped[5];
}

void MapOptimization::pointAssociateToMap(PointType const *const pi,
                                          PointType *const po) {
  float x1 = cYaw * pi->x - sYaw * pi->y;
  float y1 = sYaw * pi->x + cYaw * pi->y;
  float z1 = pi->z;

  float x2 = x1;
  float y2 = cRoll * y1 - sRoll * z1;
  float z2 = sRoll * y1 + cRoll * z1;

  po->x = cPitch * x2 + sPitch * z2 + tX;
  po->y = y2 + tY;
  po->z = -sPitch * x2 + cPitch * z2 + tZ;
  po->intensity = pi->intensity;
}

void MapOptimization::updateTransformPointCloudSinCos(PointTypePose *tIn) {
  ctRoll = cos(tIn->roll);
  stRoll = sin(tIn->roll);

  ctPitch = cos(tIn->pitch);
  stPitch = sin(tIn->pitch);

  ctYaw = cos(tIn->yaw);
  stYaw = sin(tIn->yaw);

  tInX = tIn->x;
  tInY = tIn->y;
  tInZ = tIn->z;
}

pcl::PointCloud<PointType>::Ptr MapOptimization::transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn) {
  // !!! DO NOT use pcl for point cloud transformation, results are not
  // accurate Reason: unkown
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType *pointFrom;
  PointType pointTo;

  int cloudSize = cloudIn->points.size();
  cloudOut->resize(cloudSize);

  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
    float y1 = stYaw * pointFrom->x + ctYaw * pointFrom->y;
    float z1 = pointFrom->z;

    float x2 = x1;
    float y2 = ctRoll * y1 - stRoll * z1;
    float z2 = stRoll * y1 + ctRoll * z1;

    pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
    pointTo.y = y2 + tInY;
    pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
    pointTo.intensity = pointFrom->intensity;

    cloudOut->points[i] = pointTo;
  }
  return cloudOut;
}

pcl::PointCloud<PointType>::Ptr MapOptimization::transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType *pointFrom;
  PointType pointTo;

  int cloudSize = cloudIn->points.size();
  cloudOut->resize(cloudSize);

  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    float x1 = cos(transformIn->yaw) * pointFrom->x -
               sin(transformIn->yaw) * pointFrom->y;
    float y1 = sin(transformIn->yaw) * pointFrom->x +
               cos(transformIn->yaw) * pointFrom->y;
    float z1 = pointFrom->z;

    float x2 = x1;
    float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
    float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

    pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 +
                transformIn->x;
    pointTo.y = y2 + transformIn->y;
    pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 +
                transformIn->z;
    pointTo.intensity = pointFrom->intensity;

    cloudOut->points[i] = pointTo;
  }
  return cloudOut;
}

// 发布TF变换
void MapOptimization::publishTF() {
  geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
      transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

  // 当前机器人位姿
  odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
  odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
  odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
  odomAftMapped.pose.pose.orientation.z = geoQuat.x;
  odomAftMapped.pose.pose.orientation.w = geoQuat.w;
  odomAftMapped.pose.pose.position.x = transformAftMapped[3];
  odomAftMapped.pose.pose.position.y = transformAftMapped[4];
  odomAftMapped.pose.pose.position.z = transformAftMapped[5];

  odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
  odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
  odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
  odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
  odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
  odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
  pubOdomAftMapped.publish(odomAftMapped);

  aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
  aftMappedTrans.setRotation(
      tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
  aftMappedTrans.setOrigin(tf::Vector3(
      transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
  tfBroadcaster.sendTransform(aftMappedTrans);
}


void MapOptimization::publishKeyPosesAndFrames() {
  if (pubKeyPoses.getNumSubscribers() != 0) {
    sensor_msgs::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);                                // 发布历史轨迹点
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "/camera_init";
    pubKeyPoses.publish(cloudMsgTemp);
  }

  if (pubRecentKeyFrames.getNumSubscribers() != 0) {                              // 发布当前帧的平滑特征点
    sensor_msgs::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "/camera_init";
    pubRecentKeyFrames.publish(cloudMsgTemp);
  }
}

// added by jiajia
void MapOptimization::addLaserScan(const Eigen::Vector3f& pose) {
  PointCloud scan_points;

  // 将极坐标转换为直角坐标系
  for(int i = 0; i < _scan_msg->points.size(); ++i) {
    scan_points.emplace_back(_scan_msg->points[i].x, _scan_msg->points[i].y);
  }
  
  // 定义新的scan格式，每一束光采用直角坐标
  std::shared_ptr<szyh_slam::LaserScan> laser_scan(new szyh_slam::LaserScan(scan_points));
  laser_scan->setId(_scans.size());       // 第一帧激光不做处理，仅记录并放入优化器顶点中
  // laser_scan->setPose(Eigen::Vector3f(0, 0, 0));
  laser_scan->setPose(pose);              //记录初始激光帧位置，用于slam建图初始坐标（即创建地图坐标系）
  laser_scan->transformPointCloud();      //根据激光位置，计算每个点的在map的位置
  _scans.push_back(laser_scan);           //收集每帧激光
}

std::shared_ptr<szyh_slam::ProbabilityGridMap> 
MapOptimization::getProbabilityGridMap(
    const std::vector<std::shared_ptr<szyh_slam::LaserScan>>& scans,
    double occupancy_grid_map_resolution)
{
    szyh_slam::Range map_range;
    for (const std::shared_ptr<szyh_slam::LaserScan>& scan : scans) {
        map_range.addRange(scan->getRange());
    }

    const Eigen::Vector2f& max = map_range.getMax();
    const Eigen::Vector2f& min = map_range.getMin();
    int width = ceil((max[0] - min[0]) / occupancy_grid_map_resolution);
    int height = ceil((max[1] - min[1]) / occupancy_grid_map_resolution);

    std::shared_ptr<szyh_slam::ProbabilityGridMap> probability_grid_map(new szyh_slam::ProbabilityGridMap(width, height, occupancy_grid_map_resolution));
    probability_grid_map->setOrigin(min);
    probability_grid_map->createFromScan(scans);

    return probability_grid_map;
}

void MapOptimization::publishProbabilityGridMap()
{
    auto map = getProbabilityGridMap(_scans, 0.05);
    ROS_WARN("mapping");
    nav_msgs::OccupancyGrid map_msg;
    Eigen::Vector2f origin = map->getOrigin();
    map_msg.header.stamp = ros::Time::now();
    map_msg.header.frame_id = "/map";
    map_msg.info.origin.position.x = origin.x();
    map_msg.info.origin.position.y = origin.y();
    map_msg.info.origin.orientation.x = 0;
    map_msg.info.origin.orientation.y = 0;
    map_msg.info.origin.orientation.z = 0;
    map_msg.info.origin.orientation.w = 1;
    map_msg.info.resolution = map->getResolution();
    map_msg.info.width = map->getSizeX();
    map_msg.info.height = map->getSizeY();
    map_msg.data.resize(map_msg.info.width * map_msg.info.height, -1);

    for(int i = 0; i < map_msg.data.size(); ++i) {
        int value = map->getGridValue(i);
        if(value == szyh_slam::LogOdds_Unknown) {
            map_msg.data[i] = -1;
        }
        else {
            map_msg.data[i] = map->getGridValue(i);
        }
    }

    map_pub_.publish(map_msg);
}

// 发布 3d 图
void MapOptimization::publishGlobalMap() {
  if (pubLaserCloudSurround.getNumSubscribers() == 0) return;   // 

  if (cloudKeyPoses3D->points.empty() == true) return;
  // kd-tree to find near key frames to visualize
  std::vector<int> pointSearchIndGlobalMap;                             // 
  std::vector<float> pointSearchSqDisGlobalMap;
  // search near key frames to visualize
  mtx.lock();
  kdtreeGlobalMap.setInputCloud(cloudKeyPoses3D);                       // 连续帧中，每一帧激光视角的pos x，y，z位置
  kdtreeGlobalMap.radiusSearch(
      currentRobotPosPoint, _global_map_visualization_search_radius,    // 仅提取搜索当前位置可视范围半径（默认500m）内的点
      pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);              // 
  mtx.unlock();

  for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)              // 
    globalMapKeyPoses->points.push_back(                                // 将范围内点放入全局key pose位置中
        cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
  // downsample near selected key frames
  downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
  downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);         // 降采样到1立方米内一个点，即1m一个轨迹点
  // extract visualized and downsampled key frames
  for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i) {                   // 将其机器人坐标系下的点云point，转换成世界坐标系下坐标 
    int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;     
    *globalMapKeyFrames += *transformPointCloud(
        cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);   // 包括3类，平面、角点和未分类的点云
    *globalMapKeyFrames += *transformPointCloud(
        surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);      
    *globalMapKeyFrames +=
        *transformPointCloud(outlierCloudKeyFrames[thisKeyInd],
                             &cloudKeyPoses6D->points[thisKeyInd]);
  }
  // downsample visualized points
  downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
  downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);                  // 将整个点云进行降采样，0.4立方米一个点

  sensor_msgs::PointCloud2 cloudMsgTemp;                                           // 发布 3d slam的点云地图
  pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
  cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
  cloudMsgTemp.header.frame_id = "/camera_init";
  pubLaserCloudSurround.publish(cloudMsgTemp);

  // for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i) {                   // 将其机器人坐标系下的点云point，转换成世界坐标系下坐标 
  // {
  //   pcl::PointCloud<PointType>::Ptr MapKeyFrames;
  //   MapKeyFrames.reset(new pcl::PointCloud<PointType>());

  //   MapKeyFrames->clear();
  // }

  globalMapKeyPoses->clear();
  globalMapKeyPosesDS->clear();
  globalMapKeyFrames->clear();
  //globalMapKeyFramesDS->clear();
  ROS_WARN("X:%f,    y:%f,     z:%f",currentRobotPosPoint.x,currentRobotPosPoint.y,currentRobotPosPoint.z);
  publishProbabilityGridMap();
}

// 闭环检测，找到历史中可能闭环的帧， 并用历史帧构建submap，用于闭环匹配构建新的位姿
bool MapOptimization::detectLoopClosure() {
  latestSurfKeyFrameCloud->clear();
  nearHistorySurfKeyFrameCloud->clear();
  nearHistorySurfKeyFrameCloudDS->clear();

  std::lock_guard<std::mutex> lock(mtx);
  // find the closest history key frame
  std::vector<int> pointSearchIndLoop;
  std::vector<float> pointSearchSqDisLoop;
  kdtreeHistoryKeyPoses.setInputCloud(cloudKeyPoses3D);                            // 当前机器人所在位置附近，搜索历史所有范围在闭环搜索范围内（默认为7m） 的key location，
  kdtreeHistoryKeyPoses.radiusSearch(
      currentRobotPosPoint, _history_keyframe_search_radius, pointSearchIndLoop,
      pointSearchSqDisLoop);

  closestHistoryFrameID = -1;
  for (int i = 0; i < pointSearchIndLoop.size(); ++i) {                           // 所有在闭环搜索范围的位置ID
    int id = pointSearchIndLoop[i];
    if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0) {       // 所有范围内location的时刻与现在时刻需在30s以上，否则无需闭环
      closestHistoryFrameID = id;                                                 // 仅需找到一帧曾经的位置与当前位置在 搜索范围内，且时间满足30s以上
      break;
    }
  }
  if (closestHistoryFrameID == -1) {
    return false;
  }
  // save latest key frames
  latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;                   // 获取历史key队列中最新的id
  *latestSurfKeyFrameCloud +=
      *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure],         // 将对应的特征点云转换为世界坐标系
                           &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
  *latestSurfKeyFrameCloud +=
      *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],
                           &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

  pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
  int cloudSize = latestSurfKeyFrameCloud->points.size();
  for (int i = 0; i < cloudSize; ++i) {                                          // 此帧对应的特征点云仅保留 ？？？ intensity=(float)rowIdn + (float)columnIdn / 10000.0;
    if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0) {                // 暂时觉得没用
      hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
    }
  }
  latestSurfKeyFrameCloud->clear();
  *latestSurfKeyFrameCloud = *hahaCloud;
  // save history near key frames
  for (int j = - _history_keyframe_search_num; j <= _history_keyframe_search_num; ++j) {   // 将历史中找到的一帧closestHistoryFrameID的前后_history_keyframe_search_num个作为闭环的submap
    if (closestHistoryFrameID + j < 0 ||
        closestHistoryFrameID + j > latestFrameIDLoopCloure)
      continue;
    *nearHistorySurfKeyFrameCloud += *transformPointCloud(                                 // 包括 corner 和 surf两种特征点云
        cornerCloudKeyFrames[closestHistoryFrameID + j],
        &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
    *nearHistorySurfKeyFrameCloud += *transformPointCloud(
        surfCloudKeyFrames[closestHistoryFrameID + j],
        &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
  }

  downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);              // 降采样
  downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);
  // publish history near key frames
  if (pubHistoryKeyFrames.getNumSubscribers() != 0) {                                      // 发布
    sensor_msgs::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "/camera_init";
    pubHistoryKeyFrames.publish(cloudMsgTemp);
  }

  return true;
}

// 闭环检测
void MapOptimization::performLoopClosure() {

  if (cloudKeyPoses3D->points.empty() == true)                        // 历史无location，第一次无需处理
    return;


  // try to find close key frame if there are any
  if (potentialLoopFlag == false) {
    if (detectLoopClosure() == true) {                               // 闭环查找，找到闭环和闭环使用的submap
      potentialLoopFlag = true;  // find some key frames that is old enough or
                                 // close enough for loop closure
      timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
    }
    if (potentialLoopFlag == false) return;                          // 无闭环可能无需进行闭环匹配
  }
  // reset the flag first no matter icp successes or not
  potentialLoopFlag = false;
  // ICP Settings
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(100);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);
  // Align clouds
  icp.setInputSource(latestSurfKeyFrameCloud);                        // 历史帧队列中最新的一帧的点云
  icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);                 // 闭环检测提取的submap
  pcl::PointCloud<PointType>::Ptr unused_result(
      new pcl::PointCloud<PointType>());
  icp.align(*unused_result);                                          // icp 匹配

  if (icp.hasConverged() == false || 
      icp.getFitnessScore() > _history_keyframe_fitness_score)        // 匹配结果不理想，则无需优化 
    return;
  // publish corrected cloud                                          // 发布当前点云在闭环submap中匹配后的点云
  if (pubIcpKeyFrames.getNumSubscribers() != 0) {
    pcl::PointCloud<PointType>::Ptr closed_cloud(
        new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*latestSurfKeyFrameCloud, *closed_cloud,
                             icp.getFinalTransformation());           // ICP匹配后坐标转换
    sensor_msgs::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
    cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
    cloudMsgTemp.header.frame_id = "/camera_init";
    pubIcpKeyFrames.publish(cloudMsgTemp);
  }
  /*
          get pose constraint                                         // 位置约束
          */
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionCameraFrame;
  correctionCameraFrame =
      icp.getFinalTransformation();  // get transformation in camera frame
                                     // (because points are in camera frame)
  pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch,   // 转标转换 形式改为 平移和旋转矩阵
                                    yaw);
  Eigen::Affine3f correctionLidarFrame =
      pcl::getTransformation(z, x, y, yaw, roll, pitch);                           // 坐标转换为雷达下矩阵
  // transform from world origin to wrong pose
  Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(                        // 把最新的world pose转换成lidar坐标系
      cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
  // transform from world origin to corrected pose
  Eigen::Affine3f tCorrect =
      correctionLidarFrame *
      tWrong;  // pre-multiplying -> successive rotation about a fixed frame       // 修正tf变换，为当前位置到达闭环位置的tf
  pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);          // 转换为 平移和旋转量
  gtsam::Pose3 poseFrom =
      Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));                      // 新的world位置
  gtsam::Pose3 poseTo =
      pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);        // 历史中与当前位置较近的一个位置
  gtsam::Vector Vector6(6);
  float noiseScore = icp.getFitnessScore();
  Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
      noiseScore;
  auto constraintNoise = noiseModel::Diagonal::Variances(Vector6);
  /*
          add constraints
          */
  std::lock_guard<std::mutex> lock(mtx);
  gtSAMgraph.add(
      BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID,         // 增加约束，在闭环中， 最新位置和历史位置之间的约束
                           poseFrom.between(poseTo), constraintNoise));
  isam->update(gtSAMgraph);
  isam->update();
  gtSAMgraph.resize(0);

  aLoopIsClosed = true;
}

// 提取关键帧数据
void MapOptimization::extractSurroundingKeyFrames() {
  if (cloudKeyPoses3D->points.empty() == true) return;          // 第一次无关键点时，无需提取

  // 闭环开启功能
  if (_loop_closure_enabled == true) {
    // only use recent key poses for graph building
    if (recentCornerCloudKeyFrames.size() <
        _surrounding_keyframe_search_num) {  // queue is not full (the beginning
                                         // of mapping or a loop is just
                                         // closed)
                                         // clear recent key frames queue
      recentCornerCloudKeyFrames.clear();
      recentSurfCloudKeyFrames.clear();
      recentOutlierCloudKeyFrames.clear();
      int numPoses = cloudKeyPoses3D->points.size();             // 遍历每一个轨迹点 location pose
      for (int i = numPoses - 1; i >= 0; --i) {
        int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
        updateTransformPointCloudSinCos(&thisTransformation);
        // extract surrounding map
        recentCornerCloudKeyFrames.push_front(                       // 将历史帧按最新数据的顺序放入
            transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
        recentSurfCloudKeyFrames.push_front(
            transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
        recentOutlierCloudKeyFrames.push_front(
            transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
        if (recentCornerCloudKeyFrames.size() >= _surrounding_keyframe_search_num)
          break;
      }
    } else {  // queue is full, pop the oldest key frame and push the latest  如果最近帧数据队列已满，
              // key frame
      if (latestFrameID != cloudKeyPoses3D->points.size()-1) {       // 如果机器人移动，需要剔除原来的数据，加入新的帧数据
        // if the robot is not moving, no need to
        // update recent frames

        recentCornerCloudKeyFrames.pop_front();
        recentSurfCloudKeyFrames.pop_front();
        recentOutlierCloudKeyFrames.pop_front();
        // push latest scan to the end of queue
        latestFrameID = cloudKeyPoses3D->points.size() - 1;
        PointTypePose thisTransformation =
            cloudKeyPoses6D->points[latestFrameID];
        updateTransformPointCloudSinCos(&thisTransformation);
        recentCornerCloudKeyFrames.push_back(
            transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
        recentSurfCloudKeyFrames.push_back(
            transformPointCloud(surfCloudKeyFrames[latestFrameID]));
        recentOutlierCloudKeyFrames.push_back(
            transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
      }
    }

    for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i) {     // 构建submap的点云
      *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
      *laserCloudSurfFromMap += *recentSurfCloudKeyFrames[i];
      *laserCloudSurfFromMap += *recentOutlierCloudKeyFrames[i];
    }
  } 
  else                    // 关闭闭环 
  {
    surroundingKeyPoses->clear();
    surroundingKeyPosesDS->clear();
    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses.setInputCloud(cloudKeyPoses3D);
    kdtreeSurroundingKeyPoses.radiusSearch(
        currentRobotPosPoint, (double)_surrounding_keyframe_search_radius,    // 没有闭环时，仅搜索与当前位置在50m内的位置，用于构建submap
        pointSearchInd, pointSearchSqDis);

    for (int i = 0; i < pointSearchInd.size(); ++i){
      surroundingKeyPoses->points.push_back(
          cloudKeyPoses3D->points[pointSearchInd[i]]);
    }

    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);         // 1立方米 一个location位置

    // delete key frames that are not in surrounding region
    int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
    for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {          // 遍历已经存在的location位置的ID
      bool existingFlag = false;
      for (int j = 0; j < numSurroundingPosesDS; ++j) {                       // 如果缓存中存在id 在当前位置附近50m处没有
        if (surroundingExistingKeyPosesID[i] ==
            (int)surroundingKeyPosesDS->points[j].intensity) {
          existingFlag = true;
          break;
        }
      }
      if (existingFlag == false) {                                            // 历史存储的id 不在当前50m内，则应删处此id
        surroundingExistingKeyPosesID.erase(
            surroundingExistingKeyPosesID.begin() + i);
        surroundingCornerCloudKeyFrames.erase(
            surroundingCornerCloudKeyFrames.begin() + i);
        surroundingSurfCloudKeyFrames.erase(
            surroundingSurfCloudKeyFrames.begin() + i);
        surroundingOutlierCloudKeyFrames.erase(
            surroundingOutlierCloudKeyFrames.begin() + i);
        --i;
      }
    }
    // add new key frames that are not in calculated existing key frames
    for (int i = 0; i < numSurroundingPosesDS; ++i) {                         // 若最新的location id 不在 缓存中，则应增加此ID的对应的点云数据
      bool existingFlag = false;
      for (auto iter = surroundingExistingKeyPosesID.begin();
           iter != surroundingExistingKeyPosesID.end(); ++iter) {
        if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity) {
          existingFlag = true;
          break;
        }
      }
      if (existingFlag == true) {
        continue;
      } else {
        int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
        updateTransformPointCloudSinCos(&thisTransformation);
        surroundingExistingKeyPosesID.push_back(thisKeyInd);
        surroundingCornerCloudKeyFrames.push_back(
            transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
        surroundingSurfCloudKeyFrames.push_back(
            transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
        surroundingOutlierCloudKeyFrames.push_back(
            transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
      }
    }

    for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {        // 将附近搜索范围内的n组点云形成一个整点云
      *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
      *laserCloudSurfFromMap += *surroundingSurfCloudKeyFrames[i];
      *laserCloudSurfFromMap += *surroundingOutlierCloudKeyFrames[i];
    }
  }
  // Downsample the surrounding corner key frames (or map)
  downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
  downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
  laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
  // Downsample the surrounding surf key frames (or map)
  downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
  downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
  laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
}

// 对激光点云降采样后
void MapOptimization::downsampleCurrentScan() {
  laserCloudCornerLastDS->clear();
  downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
  downSizeFilterCorner.filter(*laserCloudCornerLastDS);                 // 20立方cm 
  laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();   //角特征点云

  _scan_msgDS->clear();
  downSizeFilterCorner.setInputCloud(_scan_msg);
  downSizeFilterCorner.filter(*_scan_msgDS);                 // 20立方cm 

  laserCloudSurfLastDS->clear();
  downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
  downSizeFilterSurf.filter(*laserCloudSurfLastDS);                   // 40立方cm
  laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();      // 平面特征点云

  laserCloudOutlierLastDS->clear();
  downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
  downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);              // 40立方cm
  laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size(); // 非特征点云

  laserCloudSurfTotalLast->clear();
  laserCloudSurfTotalLastDS->clear();
  *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
  *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
  downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
  downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
  laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
}

void MapOptimization::cornerOptimization(int iterCount) {
  updatePointAssociateToMapSinCos();
  for (int i = 0; i < laserCloudCornerLastDSNum; i++) {                    // 遍历每个降采样后的角点特征点
    pointOri = laserCloudCornerLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);                             // 转换成世界坐标系
    kdtreeCornerFromMap.nearestKSearch(pointSel, 5, pointSearchInd,        // 搜索最近的5个点
                                       pointSearchSqDis);

    if (pointSearchSqDis[4] < 1.0) {                                       // 最近的5个点若都在1m内
      float cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
      }
      cx /= 5;
      cy /= 5;
      cz /= 5;

      float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
        float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
        float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

        a11 += ax * ax;
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;
      a12 /= 5;
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;

      matA1(0, 0) = a11;
      matA1(0, 1) = a12;
      matA1(0, 2) = a13;
      matA1(1, 0) = a12;
      matA1(1, 1) = a22;
      matA1(1, 2) = a23;
      matA1(2, 0) = a13;
      matA1(2, 1) = a23;
      matA1(2, 2) = a33;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> esolver(matA1);

      matD1 = esolver.eigenvalues().real();
      matV1 = esolver.eigenvectors().real();

      if (matD1[2] > 3 * matD1[1]) {
        float x0 = pointSel.x;
        float y0 = pointSel.y;
        float z0 = pointSel.z;
        float x1 = cx + 0.1 * matV1(0, 0);
        float y1 = cy + 0.1 * matV1(0, 1);
        float z1 = cz + 0.1 * matV1(0, 2);
        float x2 = cx - 0.1 * matV1(0, 0);
        float y2 = cy - 0.1 * matV1(0, 1);
        float z2 = cz - 0.1 * matV1(0, 2);

        float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                              ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                          ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                              ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                          ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                              ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

        float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                         (z1 - z2) * (z1 - z2));

        float la =
            ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
             (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
            a012 / l12;

        float lb =
            -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
              (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
            a012 / l12;

        float lc =
            -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
              (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
            a012 / l12;

        float ld2 = a012 / l12;

        float s = 1 - 0.9 * fabs(ld2);

        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        if (s > 0.1) {
          laserCloudOri->push_back(pointOri);
          coeffSel->push_back(coeff);
        }
      }
    }
  }
}

void MapOptimization::surfOptimization(int iterCount) {
  updatePointAssociateToMapSinCos();
  for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
    pointOri = laserCloudSurfTotalLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeSurfFromMap.nearestKSearch(pointSel, 5, pointSearchInd,
                                      pointSearchSqDis);

    if (pointSearchSqDis[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) =
            laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
        matA0(j, 1) =
            laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
        matA0(j, 2) =
            laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
      }
      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                 pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                 pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
                 pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

        float s = 1 - 0.9 * fabs(pd2) /
                          sqrt(sqrt(pointSel.x * pointSel.x +
                                    pointSel.y * pointSel.y +
                                    pointSel.z * pointSel.z));

        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        if (s > 0.1) {
          laserCloudOri->push_back(pointOri);
          coeffSel->push_back(coeff);
        }
      }
    }
  }
}

bool MapOptimization::LMOptimization(int iterCount) {
  float srx = sin(transformTobeMapped[0]);
  float crx = cos(transformTobeMapped[0]);
  float sry = sin(transformTobeMapped[1]);
  float cry = cos(transformTobeMapped[1]);
  float srz = sin(transformTobeMapped[2]);
  float crz = cos(transformTobeMapped[2]);

  int laserCloudSelNum = laserCloudOri->points.size();
  if (laserCloudSelNum < 50) {
    return false;
  }

  Eigen::Matrix<float,Eigen::Dynamic,6> matA(laserCloudSelNum, 6);
  Eigen::Matrix<float,6,Eigen::Dynamic> matAt(6,laserCloudSelNum);
  Eigen::Matrix<float,6,6> matAtA;
  Eigen::VectorXf matB(laserCloudSelNum);
  Eigen::Matrix<float,6,1> matAtB;
  Eigen::Matrix<float,6,1> matX;

  for (int i = 0; i < laserCloudSelNum; i++) {
    pointOri = laserCloudOri->points[i];
    coeff = coeffSel->points[i];

    float arx =
        (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
         srx * sry * pointOri.z) *
            coeff.x +
        (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) *
            coeff.y +
        (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
         cry * srx * pointOri.z) *
            coeff.z;

    float ary =
        ((cry * srx * srz - crz * sry) * pointOri.x +
         (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) *
            coeff.x +
        ((-cry * crz - srx * sry * srz) * pointOri.x +
         (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) *
            coeff.z;

    float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                 (-cry * crz - srx * sry * srz) * pointOri.y) *
                    coeff.x +
                (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                ((sry * srz + cry * crz * srx) * pointOri.x +
                 (crz * sry - cry * srx * srz) * pointOri.y) *
                    coeff.z;

    matA(i, 0) = arx;
    matA(i, 1) = ary;
    matA(i, 2) = arz;
    matA(i, 3) = coeff.x;
    matA(i, 4) = coeff.y;
    matA(i, 5) = coeff.z;
    matB(i, 0) = -coeff.intensity;
  }
  matAt = matA.transpose();
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  matX = matAtA.colPivHouseholderQr().solve(matAtB);

  if (iterCount == 0) {
    Eigen::Matrix<float,1,6> matE;
    Eigen::Matrix<float,6,6> matV;
    Eigen::Matrix<float,6,6> matV2;

    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float,6, 6> > esolver(matAtA);
    matE = esolver.eigenvalues().real();
    matV = esolver.eigenvectors().real();

     matV2 = matV;

    isDegenerate = false;
    float eignThre[6] = {100, 100, 100, 100, 100, 100};
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
    Eigen::Matrix<float,6, 1> matX2(matX);
    matX2 = matX;
    matX = matP * matX2;
  }

  transformTobeMapped[0] += matX(0, 0);
  transformTobeMapped[1] += matX(1, 0);
  transformTobeMapped[2] += matX(2, 0);
  transformTobeMapped[3] += matX(3, 0);
  transformTobeMapped[4] += matX(4, 0);
  transformTobeMapped[5] += matX(5, 0);

  float deltaR = sqrt(pow(pcl::rad2deg(matX(0, 0)), 2) +
                      pow(pcl::rad2deg(matX(1, 0)), 2) +
                      pow(pcl::rad2deg(matX(2, 0)), 2));
  float deltaT = sqrt(pow(matX(3, 0) * 100, 2) +
                      pow(matX(4, 0) * 100, 2) +
                      pow(matX(5, 0) * 100, 2));

  if (deltaR < 0.05 && deltaT < 0.05) {
    return true;
  }
  return false;
}

// scan对map匹配优化
void MapOptimization::scan2MapOptimization() {
  if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {   // 如果当前激光帧特征点个数达到一定个数时，才进行匹配优化
    kdtreeCornerFromMap.setInputCloud(laserCloudCornerFromMapDS);
    kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapDS);

    for (int iterCount = 0; iterCount < 10; iterCount++) {                       // 优化匹配迭代10次，
      laserCloudOri->clear();
      coeffSel->clear();

      cornerOptimization(iterCount);
      surfOptimization(iterCount);

      if (LMOptimization(iterCount) == true) break;
    }

    transformUpdate();                                                           // 获取匹配后当前帧的位姿 
  }
}

// 记录当前帧的位姿，按时间顺序存储队列中
void MapOptimization::saveKeyFramesAndFactor() {
  currentRobotPosPoint.x = transformAftMapped[3];                                // 当前帧的pose
  currentRobotPosPoint.y = transformAftMapped[4];
  currentRobotPosPoint.z = transformAftMapped[5];

  gtsam::Vector Vector6(6);
  Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
  auto priorNoise = noiseModel::Diagonal::Variances(Vector6);
  auto odometryNoise = noiseModel::Diagonal::Variances(Vector6);

  bool saveThisKeyFrame = true;
  if (sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x) *                   // 帧移动的位置在0.3间隔进行保存
               (previousRobotPosPoint.x - currentRobotPosPoint.x) +
           (previousRobotPosPoint.y - currentRobotPosPoint.y) *
               (previousRobotPosPoint.y - currentRobotPosPoint.y) +
           (previousRobotPosPoint.z - currentRobotPosPoint.z) *
               (previousRobotPosPoint.z - currentRobotPosPoint.z)) < 0.3) {
    saveThisKeyFrame = false;
  }

  if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty()) return;     // 不够0.3m位移 无需处理

  previousRobotPosPoint = currentRobotPosPoint;
  /**
   * update grsam graph
   */
  if (cloudKeyPoses3D->points.empty()) {                                        // 
    // 第一个约束条件， 优化时可固定不变
    gtSAMgraph.add(PriorFactor<Pose3>(
        0,
        Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0],
                           transformTobeMapped[1]),
              Point3(transformTobeMapped[5], transformTobeMapped[3],
                     transformTobeMapped[4])),
        priorNoise));
    // 第一个节点
    initialEstimate.insert(                                                     // 
        0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0],
                              transformTobeMapped[1]),
                 Point3(transformTobeMapped[5], transformTobeMapped[3],
                        transformTobeMapped[4])));
    for (int i = 0; i < 6; ++i) transformLast[i] = transformTobeMapped[i];
    
    // // added by jiajia
    // addLaserScan(Eigen::Vector3f(currentRobotPosPoint.x,currentRobotPosPoint.y,transformAftMapped[2])); 
  } 
  else
  {
    // 上时刻的位姿
    gtsam::Pose3 poseFrom = Pose3(
        Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
        Point3(transformLast[5], transformLast[3], transformLast[4]));             

    // 当前时刻的位姿
    gtsam::Pose3 poseTo =
        Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
                           transformAftMapped[1]),
              Point3(transformAftMapped[5], transformAftMapped[3],
                     transformAftMapped[4]));

    // 增加一个约束条件
    gtSAMgraph.add(BetweenFactor<Pose3>(                                      
        cloudKeyPoses3D->points.size() - 1, cloudKeyPoses3D->points.size(),
        poseFrom.between(poseTo), odometryNoise));

    // 增加一个节点
    initialEstimate.insert(                                                    
        cloudKeyPoses3D->points.size(),
        Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
                           transformAftMapped[1]),
              Point3(transformAftMapped[5], transformAftMapped[3],
                     transformAftMapped[4])));
  }
  /**
   * update iSAM
   */
  isam->update(gtSAMgraph, initialEstimate);
  isam->update();

  gtSAMgraph.resize(0);
  initialEstimate.clear();

  /**
   * save key poses
   */
  PointType thisPose3D;
  PointTypePose thisPose6D;
  Pose3 latestEstimate;

  isamCurrentEstimate = isam->calculateEstimate();
  latestEstimate =
      isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);   //优化后定位

  thisPose3D.x = latestEstimate.translation().y();
  thisPose3D.y = latestEstimate.translation().z();
  thisPose3D.z = latestEstimate.translation().x();
  thisPose3D.intensity =
      cloudKeyPoses3D->points.size();                     // this can be used as index, 历史key location pose中的intensity为存储顺序的ID
  cloudKeyPoses3D->push_back(thisPose3D);                 // 记录每帧激光点所在的位置 3d

  thisPose6D.x = thisPose3D.x;
  thisPose6D.y = thisPose3D.y;
  thisPose6D.z = thisPose3D.z;
  thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
  thisPose6D.roll = latestEstimate.rotation().pitch();
  thisPose6D.pitch = latestEstimate.rotation().yaw();
  thisPose6D.yaw = latestEstimate.rotation().roll();  // in camera frame
  thisPose6D.time = timeLaserOdometry;
  cloudKeyPoses6D->push_back(thisPose6D);                 // 记录每帧激光雷达所在位姿 6d
  /**
   * save updated transform
   */
  if (cloudKeyPoses3D->points.size() > 1) {
    transformAftMapped[0] = latestEstimate.rotation().pitch();
    transformAftMapped[1] = latestEstimate.rotation().yaw();
    transformAftMapped[2] = latestEstimate.rotation().roll();
    transformAftMapped[3] = latestEstimate.translation().y();
    transformAftMapped[4] = latestEstimate.translation().z();
    transformAftMapped[5] = latestEstimate.translation().x();

    for (int i = 0; i < 6; ++i) {
      transformLast[i] = transformAftMapped[i];
      transformTobeMapped[i] = transformAftMapped[i];
    }
  }

  pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(
      new pcl::PointCloud<PointType>());

  pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
  pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
  pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

  cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
  surfCloudKeyFrames.push_back(thisSurfKeyFrame);
  outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);

  // added by jiajia
  addLaserScan(Eigen::Vector3f(thisPose6D.z,thisPose6D.x,thisPose6D.pitch)); 
}

// 若存在闭环处理，则需要对位姿进行修正，将历史的的位姿用优化后的数据进行更新
void MapOptimization::correctPoses() {
  if (aLoopIsClosed == true) {
    recentCornerCloudKeyFrames.clear();
    recentSurfCloudKeyFrames.clear();
    recentOutlierCloudKeyFrames.clear();
    // update key poses
    int numPoses = isamCurrentEstimate.size();
    for (int i = 0; i < numPoses; ++i) {
      cloudKeyPoses3D->points[i].x =
          isamCurrentEstimate.at<Pose3>(i).translation().y();
      cloudKeyPoses3D->points[i].y =
          isamCurrentEstimate.at<Pose3>(i).translation().z();
      cloudKeyPoses3D->points[i].z =
          isamCurrentEstimate.at<Pose3>(i).translation().x();

      cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
      cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
      cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
      cloudKeyPoses6D->points[i].roll =
          isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
      cloudKeyPoses6D->points[i].pitch =
          isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
      cloudKeyPoses6D->points[i].yaw =
          isamCurrentEstimate.at<Pose3>(i).rotation().roll();
      _scans[i]->setPose(Eigen::Vector3f(cloudKeyPoses6D->points[i].x,cloudKeyPoses6D->points[i].y,cloudKeyPoses6D->points[i].yaw));
      _scans[i]->transformPointCloud();
    }

    aLoopIsClosed = false;                  // 修正完成
  }
}

void MapOptimization::clearCloud() {
  laserCloudCornerFromMap->clear();
  laserCloudSurfFromMap->clear();
  laserCloudCornerFromMapDS->clear();
  laserCloudSurfFromMapDS->clear();
}


void MapOptimization::run() {
  size_t cycle_count = 0;

  while (ros::ok()) {
    AssociationOut association;
    _input_channel.receive(association);
    if( !ros::ok() ) break;

    {
      std::lock_guard<std::mutex> lock(mtx);
      // added by jiajia
      _scan_msg = association.scan_msg;
      laserCloudCornerLast = association.cloud_corner_last;
      laserCloudSurfLast = association.cloud_surf_last;
      laserCloudOutlierLast = association.cloud_outlier_last;

      timeLaserOdometry = association.laser_odometry.header.stamp.toSec();
      timeLastProcessing = timeLaserOdometry;

      OdometryToTransform(association.laser_odometry, transformSum);   // 4元数转换为旋转量

      transformAssociateToMap();                                       // 坐标转为世界坐标系

      extractSurroundingKeyFrames();                                   // 提取关键数据点云帧，主要根据当前位置附近搜索范围的位置对应的点云，形成submap，闭环采用固定个数帧，非闭环存储附近一点距离内的点云帧

      downsampleCurrentScan();                                         // 对当前帧进行降采样

      scan2MapOptimization();                                          // 用当前帧对submap进行匹配

      saveKeyFramesAndFactor();                                        // 对新的一帧点云及其对应位置进行保存

      correctPoses();                                                  // 出现闭环时，优化的位姿，对位姿进行更新修正

      publishTF();

      publishKeyPosesAndFrames();                                      // 发布轨迹点keypose 

      clearCloud();
    }
    cycle_count++;

    if ((cycle_count % 3) == 0) {
      _loop_closure_signal.send(true);
    }

    if ((cycle_count % 10) == 0) {
      _publish_global_signal.send(true);
    }
  }
}
