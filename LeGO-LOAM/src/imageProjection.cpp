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

#include <boost/circular_buffer.hpp>
#include "imageProjection.h"

ImageProjection::ImageProjection(ros::NodeHandle& nh,
                                 Channel<ProjectionOut>& output_channel)
    : _nh(nh),
      _output_channel(output_channel)
{
  _sub_laser_cloud = nh.subscribe<sensor_msgs::PointCloud2>(         //接收激光topic，topic可映射
      "/lidar_points", 1, &ImageProjection::cloudHandler, this);

  _pub_full_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
  _pub_full_info_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

  _pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
  _pub_segmented_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
  _pub_segmented_cloud_pure =
      nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud_pure", 1);
  _pub_segmented_cloud_info =
      nh.advertise<cloud_msgs::cloud_info>("/segmented_cloud_info", 1);
  _pub_outlier_cloud = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);

  // added by wangjiajia
  _pub_laser = nh.advertise<sensor_msgs::PointCloud2>("/scan", 1);

  nh.getParam("/lego_loam/laser/num_vertical_scans", _vertical_scans);
  nh.getParam("/lego_loam/laser/num_horizontal_scans", _horizontal_scans);
  nh.getParam("/lego_loam/laser/vertical_angle_bottom", _ang_bottom);
  float vertical_angle_top;
  nh.getParam("/lego_loam/laser/vertical_angle_top", vertical_angle_top);

  // 雷达的属性，包括水平和垂直角度分辨率、垂直方向起始角度
  _ang_resolution_X = (M_PI*2) / (_horizontal_scans);
  _ang_resolution_Y = DEG_TO_RAD*(vertical_angle_top - _ang_bottom) / float(_vertical_scans-1);
  _ang_bottom = -( _ang_bottom - 0.1) * DEG_TO_RAD;                          // 垂直方向上起始角度
  _segment_alpha_X = _ang_resolution_X;
  _segment_alpha_Y = _ang_resolution_Y;

  nh.getParam("/lego_loam/imageProjection/segment_theta", _segment_theta);   // 分割的角度
  _segment_theta *= DEG_TO_RAD;

  nh.getParam("/lego_loam/imageProjection/segment_valid_point_num",
              _segment_valid_point_num);
  nh.getParam("/lego_loam/imageProjection/segment_valid_line_num",
              _segment_valid_line_num);

  nh.getParam("/lego_loam/laser/ground_scan_index",
              _ground_scan_index);                                    // 垂直方向搜索地面点云的垂直方向最大索引，
                                                                      // 如7，由于hdl16为16根线，则0~7即水平以下可能测到地面

  nh.getParam("/lego_loam/laser/sensor_mount_angle",
              _sensor_mount_angle);
  _sensor_mount_angle *= DEG_TO_RAD;

  const size_t cloud_size = _vertical_scans * _horizontal_scans;      // 激光点云总个数
                                                                      // 实例化所有点云
  _laser_cloud_in.reset(new pcl::PointCloud<PointType>());            // 输入的原始激光点云
  _full_cloud.reset(new pcl::PointCloud<PointType>());                // 根据激光的参数，一帧全部的点云数据包括无效值
  _full_info_cloud.reset(new pcl::PointCloud<PointType>());           // 包含tag信息的的点云数据

  _ground_cloud.reset(new pcl::PointCloud<PointType>());              // 分割后地面的点云 
  _segmented_cloud.reset(new pcl::PointCloud<PointType>());
  _segmented_cloud_pure.reset(new pcl::PointCloud<PointType>());
  _outlier_cloud.reset(new pcl::PointCloud<PointType>());

  // added by jiajia
  _scan_msg.reset(new pcl::PointCloud<PointType>());

  // 激光原始点云数目提前设置
  _full_cloud->points.resize(cloud_size);                             // 完整的点云信息个数为固定
  _full_info_cloud->points.resize(cloud_size);

}

// 初始化所有参数
void ImageProjection::resetParameters() {
  const size_t cloud_size = _vertical_scans * _horizontal_scans;
  PointType nanPoint;
  nanPoint.x = std::numeric_limits<float>::quiet_NaN();    //应该是等于NAN
  nanPoint.y = std::numeric_limits<float>::quiet_NaN();
  nanPoint.z = std::numeric_limits<float>::quiet_NaN();

  _laser_cloud_in->clear();                                //中间buffer点云全部默认
  _ground_cloud->clear();
  _segmented_cloud->clear();
  _segmented_cloud_pure->clear();
  _outlier_cloud->clear();

  _range_mat.resize(_vertical_scans, _horizontal_scans);   //定义激光点的个数，用于存储距离
  _ground_mat.resize(_vertical_scans, _horizontal_scans);  // 存储地面
  _label_mat.resize(_vertical_scans, _horizontal_scans);   // 带有tag的点云 

  _range_mat.fill(FLT_MAX);                                // 全部初始化最大值
  _ground_mat.setZero();                                   // 初始化为0
  _label_mat.setZero();                                    // 初始化为0

  _label_count = 1;                                        // 点云分类后label的第一个label标识号
                                                           // 全点云预设无效值
  std::fill(_full_cloud->points.begin(), _full_cloud->points.end(), nanPoint);
  std::fill(_full_info_cloud->points.begin(), _full_info_cloud->points.end(),
            nanPoint);

  _seg_msg.startRingIndex.assign(_vertical_scans, 0);      // 16线起始索引均初始化为0， 采用容器的方式， 第一参数表明size， 第二参数表明 value
  _seg_msg.endRingIndex.assign(_vertical_scans, 0);        // 同上

  _seg_msg.segmentedCloudGroundFlag.assign(cloud_size, false);
  _seg_msg.segmentedCloudColInd.assign(cloud_size, 0);
  _seg_msg.segmentedCloudRange.assign(cloud_size, 0);

  // added by wangjiajia 
  _scan_msg->clear();
  // _scan_msg.range_min = 0.1;
  // _scan_msg.range_max = 80;
  // _scan_msg.angle_min = -M_PI;
  // _scan_msg.angle_max = M_PI;
  // _scan_msg.angle_increment = 2*M_PI/_horizontal_scans;
  // _scan_msg.ranges.resize(_horizontal_scans);        //垂直投影为2维laser
  // _scan_msg.intensities.resize(_horizontal_scans);   // 
}

// 接收激光点云处理回调
void ImageProjection::cloudHandler(
    const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
  // Reset parameters
  resetParameters();

  // Copy and remove NAN points
  pcl::fromROSMsg(*laserCloudMsg, *_laser_cloud_in);                          // ros message 转换为 pcl point
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_laser_cloud_in, *_laser_cloud_in, indices);  // 剔除非正常数
  _seg_msg.header = laserCloudMsg->header;

  // 具体功能还未理解，感觉就是搜索一个雷达第一个点起始角度和最后一点的角度，在水平方向上
  findStartEndAngle();
  // Range image projection  ， 将无序点云，变为有序点云进行存储，包括水平和垂直索引号
  projectPointCloud();
  // Mark ground points， 将所有点云进行地面分割，提取地面数据和非地面数据
  groundRemoval();
  // Point cloud segmentation ， 点云进行聚类，非地面数据用label进行标记
  cloudSegmentation();
  //publish (optionally)  ，发布topic 
  publishClouds();
}


// 将输入的原始点云还原成激光扫描方式的排列，形成2维深度图，即V×H大小2维矩阵
void ImageProjection::projectPointCloud() {
  // range image projection
  const size_t cloudSize = _laser_cloud_in->points.size();

  for (size_t i = 0; i < cloudSize; ++i) {                             // 遍历每一个点云信息
    PointType thisPoint = _laser_cloud_in->points[i];

    float range = sqrt(thisPoint.x * thisPoint.x +                     // 反推点的距离
                       thisPoint.y * thisPoint.y +
                       thisPoint.z * thisPoint.z);

    // find the row and column index in the image for this point       
    float verticalAngle = std::asin(thisPoint.z / range);              // 获取z轴的角度，用于获取在16线中的哪一索引
        //std::atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y));

    int rowIdn = (verticalAngle + _ang_bottom) / _ang_resolution_Y;    // 获取 扫描线中的索引号，因为前面_ang_bottom取过反，故采用加法(??不知前方为何取反)
    if (rowIdn < 0 || rowIdn >= _vertical_scans) {
      continue;
    }

    float horizonAngle = std::atan2(thisPoint.x, thisPoint.y);         // x/y ，范围为-PI～ PI， pi/2 表明为x轴方向

    int columnIdn = -round((horizonAngle - M_PI_2) / _ang_resolution_X) + _horizontal_scans * 0.5;   //  ??? 不知道为什么这么绕。 表明x轴方向为中间索引号

    if (columnIdn >= _horizontal_scans){
      columnIdn -= _horizontal_scans;
    }

    if (columnIdn < 0 || columnIdn >= _horizontal_scans){
      continue;
    }

    if (range < 0.1){                                                  // 距离过小忽略
      continue;
    }

    _range_mat(rowIdn, columnIdn) = range;                            // 矩阵中填充距离信息，如此可提取16线中每条线的数据，每条线按顺序存储，可当做单线雷达

    thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0; // 强度根据不同层则不一样，可用于显示区分

    size_t index = columnIdn + rowIdn * _horizontal_scans;            // 索引号
    _full_cloud->points[index] = thisPoint;                           // 存入两个点云中，点云数据包含每个点的在16线的信息
    // the corresponding range of a point is saved as "intensity"
    _full_info_cloud->points[index] = thisPoint;
    _full_info_cloud->points[index].intensity = range;                // 强度信息是距离信息
  }
}

// 查找整个点云数据起始角度和终止角度
// 应该是查找整个点云的起始和终止角度，或者说激光雷达开始扫描和结束的水平方向角度
// 其目的是可以识别任何型号的3d雷达，而非直接原始雷达本身数据
// ??? 有点不太懂的是，原始点云均是相对于激光雷达为原点坐标。理论上其起始扫描角度和终止扫描角度应该是雷达参数决定。
// 将查资料可得出是因为vlp激光雷达驱动导致的，和2d雷达不一样，雷达输出距离和对应的角度，故角度并非参数推算出来，而是直接获取的
// 由于测量的时序问题，其不能保证每次的起点和终点都是确定的，所以较为复杂和难懂
// 其目的应该是可不关心雷达型号和参数，自行获取。
// 但是不知为何需要将起始和终止角度全部加个负号（即镜像），猜测估计和雷达旋转方向有关。大多数逆时针旋转，估计如此可改为顺时针。
void ImageProjection::findStartEndAngle() {
  // start and end orientation of this cloud
  auto point = _laser_cloud_in->points.front();
  _seg_msg.startOrientation = -std::atan2(point.y, point.x);                // ??? 根据点云计算激光雷达起始扫描角度

  point = _laser_cloud_in->points.back();
  _seg_msg.endOrientation = -std::atan2(point.y, point.x) + 2 * M_PI;       // ??? 终止角度 + 360 

  if (_seg_msg.endOrientation - _seg_msg.startOrientation > 3 * M_PI) {     // 起始角度和终止角度 放在0~360之间
    _seg_msg.endOrientation -= 2 * M_PI;
  } else if (_seg_msg.endOrientation - _seg_msg.startOrientation < M_PI) {
    _seg_msg.endOrientation += 2 * M_PI;
  }
  _seg_msg.orientationDiff =                                                 // 终止与起始角度差
      _seg_msg.endOrientation - _seg_msg.startOrientation;
}

//提取出地面数据放入_ground_cloud中
// _ground_mat中为1也为地面
// _label_mat 为0则为非地面
void ImageProjection::groundRemoval() {
  // _ground_mat
  // -1, no valid info to check if ground of not
  //  0, initial value, after validation, means not ground
  //  1, ground
  for (size_t j = 0; j < _horizontal_scans; ++j) {
    for (size_t i = 0; i < _ground_scan_index; ++i) {           // 仅遍历_ground_scan_index，即水平以下的8跟扫描线
      size_t lowerInd = j + (i)*_horizontal_scans;
      size_t upperInd = j + (i + 1) * _horizontal_scans;

      if (_full_cloud->points[lowerInd].intensity == -1 ||      // 垂直方向上相邻的两个点有一个存在无效值，？？？？？？？？没看到哪里赋值为无效值,不起任何作用
          _full_cloud->points[upperInd].intensity == -1) {
        // no info to check, invalid points
        _ground_mat(i, j) = -1;                                 // 表明此点无法判断
        continue;
      }

      float dX =
          _full_cloud->points[upperInd].x - _full_cloud->points[lowerInd].x;
      float dY =
          _full_cloud->points[upperInd].y - _full_cloud->points[lowerInd].y;
      float dZ =
          _full_cloud->points[upperInd].z - _full_cloud->points[lowerInd].z;

      float vertical_angle = std::atan2(dZ , sqrt(dX * dX + dY * dY + dZ * dZ));               // 存在bug，我觉的应该是sqrt(dX * dX + dY * dY）

      // TODO: review this change, 判断前后两点的角度变化在10度内

      if ( (vertical_angle - _sensor_mount_angle) <= 10 * DEG_TO_RAD) {
        _ground_mat(i, j) = 1;
        _ground_mat(i + 1, j) = 1;
      }
    }
  }
  // extract ground cloud (_ground_mat == 1)
  // mark entry that doesn't need to label (ground and invalid point) for
  // segmentation note that ground remove is from 0~_N_scan-1, need _range_mat
  // for mark label matrix for the 16th scan
  // 标记地面数据和无效数据 为-1， 即默认0 则为非地面有效数据
  for (size_t i = 0; i < _vertical_scans; ++i) {
    for (size_t j = 0; j < _horizontal_scans; ++j) {
      if (_ground_mat(i, j) == 1 ||
          _range_mat(i, j) == FLT_MAX) {
        _label_mat(i, j) = -1;    
      }
    }
  }

  // 记录地面点云数据
  for (size_t i = 0; i <= _ground_scan_index; ++i) {
    for (size_t j = 0; j < _horizontal_scans; ++j) {
      if (_ground_mat(i, j) == 1)
        _ground_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
    }
  }

  // 投影成2维激光
  #if 1
  for (size_t j = 0; j < _horizontal_scans; ++j) {
    float min_range = 1000;
    size_t id_min = 0;
    for (size_t i = 0; i < _vertical_scans; ++i) {
      size_t Ind = j + (i)*_horizontal_scans;
      float Z = _full_cloud->points[Ind].z;
      if ((_ground_mat(i, j) != 1) &&
          (Z > 0.4) && (Z<1.2) &&
          (_range_mat(i, j)<40)) {                     // 地面上点云忽略, 过高过矮的点忽略， 过远的点忽略
          if(_range_mat(i, j) < min_range) {           // 计算最小距离
            min_range = _range_mat(i, j);
            id_min = Ind;
          }
      }
    }
    if (min_range<1000) {
      _scan_msg->push_back(_full_cloud->points[id_min]);
    }
  }
  #else
  for (size_t j = 0; j < _horizontal_scans; ++j) {
    size_t Ind = j + (7)*_horizontal_scans;
    float Z = _full_cloud->points[Ind].z;
    if ((_ground_mat(7, j) != 1) &&
        (_range_mat(7, j)<80)) {                     // 地面上点云忽略, 过高过矮的点忽略， 过远的点忽略
          _scan_msg.ranges[j] = _range_mat(7, j);
    }
    else {
      _scan_msg.ranges[j] = std::numeric_limits<float>::infinity();
    }
  }
  #endif

  //
}

// 非地面的有效点云分类
// _label_mat 初始为0 非地面有效点云
// _label_mat 初始为-1 则无需考虑
// 分类后 _label_mat中为分类label标志
void ImageProjection::cloudSegmentation() {
  // segmentation process ， 仅处理未被分类的数据，即标记0的点云， 完成分类获取_label_mat
  for (size_t i = 0; i < _vertical_scans; ++i)
    for (size_t j = 0; j < _horizontal_scans; ++j)
      if (_label_mat(i, j) == 0) labelComponents(i, j);

  int sizeOfSegCloud = 0;
  // extract segmented cloud for lidar odometry， 提取分割的点云用于激光里程计使用
  for (size_t i = 0; i < _vertical_scans; ++i) {
    _seg_msg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

    for (size_t j = 0; j < _horizontal_scans; ++j) {
      if (_label_mat(i, j) > 0 || _ground_mat(i, j) == 1) {    // 标记过和地面上的数据，都需要判断
        // outliers that will not be used for optimization (always continue)
        if (_label_mat(i, j) == 999999) {                      // 无用的类处理（即类聚点数过少的估计点云簇）
          if (i > _ground_scan_index && j % 5 == 0) {          // 但是垂直索引如果高于地面索引的（即非地面上的点），每隔5个点（降采样）放入_outlier_cloud点云
            _outlier_cloud->push_back(
                _full_cloud->points[j + i * _horizontal_scans]);
            continue;                                          // 小点云簇降采样放入_outlier_cloud中
          } else {
            continue;
          }
        }
        // majority of ground points are skipped， 若是地面上点，则应每隔5点，中间则忽略
        if (_ground_mat(i, j) == 1) {
          if (j % 5 != 0 && j > 5 && j < _horizontal_scans - 5) continue;
        }
        // mark ground points so they will not be considered as edge features
        // later
        _seg_msg.segmentedCloudGroundFlag[sizeOfSegCloud] =    //记录此index是否为地面点
            (_ground_mat(i, j) == 1);
        // mark the points' column index for marking occlusion later
        _seg_msg.segmentedCloudColInd[sizeOfSegCloud] = j;     //记录此index的水平索引号
        // save range info
        _seg_msg.segmentedCloudRange[sizeOfSegCloud] =         //记录距离
            _range_mat(i, j);
        // save seg cloud                                      //记录点云降采样后的地面及其有效分类的点云
        _segmented_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
        // size of seg cloud
        ++sizeOfSegCloud;
      }
    }

    _seg_msg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;          // 起始和终止边界，并剔除两侧5个点
  }

  // extract segmented cloud for visualization                  // 聚类后的结果用于显示， 地面和无效的不显示
  for (size_t i = 0; i < _vertical_scans; ++i) {
    for (size_t j = 0; j < _horizontal_scans; ++j) {
      if (_label_mat(i, j) > 0 && _label_mat(i, j) != 999999) {
        _segmented_cloud_pure->push_back(
            _full_cloud->points[j + i * _horizontal_scans]);
        _segmented_cloud_pure->points.back().intensity =
            _label_mat(i, j);
      }
    }
  }
}

// 分类标记，采用连通扩展标记法，即判断当前点，然后不断扩展相连区域，相邻区域若为同一类别，继续
void ImageProjection::labelComponents(int row, int col) {

  const float segmentThetaThreshold = tan(_segment_theta);        // 斜度阈值，用于判断一个平面阈值

  std::vector<bool> lineCountFlag(_vertical_scans, false);        // 定义16个 bool 向量状态
  const size_t cloud_size = _vertical_scans * _horizontal_scans;
  using Coord2D = Eigen::Vector2i;
  boost::circular_buffer<Coord2D> queue(cloud_size);              // 构建循环容器，大小为整个点云个数
  boost::circular_buffer<Coord2D> all_pushed(cloud_size);

  queue.push_back({ row,col } );                                  // 用于遍历分类
  all_pushed.push_back({ row,col } );

  const Coord2D neighborIterator[4] = {
      {0, -1}, {-1, 0}, {1, 0}, {0, 1}};

  while (queue.size() > 0) {                                      // 每次循环将queue中每个元素进行判断
    // Pop point 
    Coord2D fromInd = queue.front(); 
    queue.pop_front();

    // Mark popped point
    _label_mat(fromInd.x(), fromInd.y()) = _label_count;          // 遍历标记为_label_count
    // Loop through all the neighboring grids of popped grid

    for (const auto& iter : neighborIterator) {                   // 判断上下左右4个点云数据
      // new index
      int thisIndX = fromInd.x() + iter.x();
      int thisIndY = fromInd.y() + iter.y();
      // index should be within the boundary
      if (thisIndX < 0 || thisIndX >= _vertical_scans){           // 获取索引，并忽略垂直方向上的界外索引
        continue;
      }
      // at range image margin (left or right side)               // 水平方向是循环的，因此需要考虑循环的索引
      if (thisIndY < 0){
        thisIndY = _horizontal_scans - 1;
      }
      if (thisIndY >= _horizontal_scans){
        thisIndY = 0;
      }
      // prevent infinite loop (caused by put already examined point back)  // 已分类的，无需再次判断
      if (_label_mat(thisIndX, thisIndY) != 0){
        continue;
      }

      float d1 = std::max(_range_mat(fromInd.x(), fromInd.y()),    // 获取当前点和相邻点，距离较大值
                    _range_mat(thisIndX, thisIndY));
      float d2 = std::min(_range_mat(fromInd.x(), fromInd.y()),    // 获取较小值
                    _range_mat(thisIndX, thisIndY));

      float alpha = (iter.x() == 0) ? _ang_resolution_X : _ang_resolution_Y;  // 根据相邻方向获取水平或垂直方向的角度分辨率
      float tang = (d2 * sin(alpha) / (d1 - d2 * cos(alpha)));     // 实际为短线向长线做垂直线， 长线端点离垂线位置，越近，表明越平坦

      if (tang > segmentThetaThreshold) {                          // 越大表明越平坦，表明为同一分类，放入queue，继续扩展分类
        queue.push_back( {thisIndX, thisIndY } );

        _label_mat(thisIndX, thisIndY) = _label_count;             // 将其标记为同一分类
        lineCountFlag[thisIndX] = true;                            // 垂直方向的此行，已分类过

        all_pushed.push_back(  {thisIndX, thisIndY } );            // 是此分类的，均放入放入all pushed
      }
    }
  }                                                                // 以上代码，完成了某一点连续扩展，直到分类结束

  // check if this segment is valid
  bool feasibleSegment = false;                                    // 分类的点云大于30点以上，则有效
  if (all_pushed.size() >= 30){
    feasibleSegment = true;
  }
  else if (all_pushed.size() >= _segment_valid_point_num) {        // 如果分类个数大于 _segment_valid_point_num， 
    int lineCount = 0;                                             // 并且在垂直方向的个数大于_segment_valid_line_num，则也应为 有效分类
    for (size_t i = 0; i < _vertical_scans; ++i) {
      if (lineCountFlag[i] == true) ++lineCount;
    }
    if (lineCount >= _segment_valid_line_num) feasibleSegment = true;
  }
  // segment is valid, mark these points ,                         如果此分类有效，则将标记+1， 用于标记下一个分类
  if (feasibleSegment == true) {
    ++_label_count;
  } else {  // segment is invalid, mark these points，              否则将label矩阵对应位置标记为无用类  
    for (size_t i = 0; i < all_pushed.size(); ++i) {
      _label_mat(all_pushed[i].x(), all_pushed[i].y()) = 999999;
    }
  }
}

// added by wangjiajia
//
// void ImageProjection::publishlaser(sensor_msgs::PointCloud2& temp) {
//   _scan_msg.header.stamp = temp.header.stamp;
//   _scan_msg.header.frame_id = "base_link";
//   _pub_laser.publish(_scan_msg);
// }

void ImageProjection::publishClouds() {

  sensor_msgs::PointCloud2 temp;
  temp.header.stamp = _seg_msg.header.stamp;
  temp.header.frame_id = "base_link";

  auto PublishCloud = [](ros::Publisher& pub, sensor_msgs::PointCloud2& temp,     
                          const pcl::PointCloud<PointType>::Ptr& cloud) {
    if (pub.getNumSubscribers() != 0) {
      pcl::toROSMsg(*cloud, temp);
      temp.header.frame_id = "base_link";                                         // 修复，源代码有bug，需增加framid
      pub.publish(temp);
    }
  };

  PublishCloud(_pub_outlier_cloud, temp, _outlier_cloud);                         // 没有被准确分类的点云数据,且每隔5点进行抽样
  PublishCloud(_pub_segmented_cloud, temp, _segmented_cloud);                     // 聚类的结果，包括地面信息，但不包括未分类的孤立的点云，其中地面信息每隔5点采样
  PublishCloud(_pub_full_cloud, temp, _full_cloud);                               // 全部点云信息，剔除过近的点，强度为不同垂直的索引
  PublishCloud(_pub_ground_cloud, temp, _ground_cloud);                           // 仅包含地面点云，且不同行，强度信息有区分
  PublishCloud(_pub_segmented_cloud_pure, temp, _segmented_cloud_pure);           // 聚类的结果，不包括地面和一些孤立点云簇（即个数过少），但包含垂直方向上点跨过3行5个点数以上的数据
  PublishCloud(_pub_full_info_cloud, temp, _full_info_cloud);                     // 全部点云信息，剔除过近的点，强度为距离
  //added by jiajia
  PublishCloud(_pub_laser, temp, _scan_msg);  

  if (_pub_segmented_cloud_info.getNumSubscribers() != 0) {
    _pub_segmented_cloud_info.publish(_seg_msg);                                  //_segmented_cloud 与 _seg_msg 内容一致，包含地面信息的聚类的点
  }                                                                               // 但是以1维数组进行存储，同时记录了没行数据在一维数组中起始位置
 
  //added by wangjiajia
  // publishlaser(temp);

  //--------------------
  ProjectionOut out;                                                              // 管道输出包括三种点云
  out.outlier_cloud.reset(new pcl::PointCloud<PointType>());
  out.segmented_cloud.reset(new pcl::PointCloud<PointType>());
  out.scan_msg.reset(new pcl::PointCloud<PointType>()); 
  std::swap( out.seg_msg, _seg_msg);                                              // 交换两个数，实际也为赋值
  std::swap(out.outlier_cloud, _outlier_cloud);
  std::swap(out.segmented_cloud, _segmented_cloud);
  std::swap(out.scan_msg, _scan_msg);

  _output_channel.send( std::move(out) );

}


