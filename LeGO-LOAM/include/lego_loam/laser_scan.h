#ifndef LASER_SCAN_H
#define LASER_SCAN_H

#include <iostream>
#include <vector>
#include <Eigen/Geometry>

typedef std::vector<Eigen::Vector2f> PointCloud;

namespace szyh_slam
{

class Range
{
public:
    Range() : min_(std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
              max_(std::numeric_limits<float>::min(), std::numeric_limits<float>::min()) {}
    void addPoint(const Eigen::Vector2f& point)
    {
        if(point[0] < min_[0]) {
            min_[0] = point[0];
        }
        else if(point[0] > max_[0]) {
            max_[0] = point[0];
        }

        if(point[1] < min_[1]) {
            min_[1] = point[1];
        }
        else if(point[1] > max_[1]) {
            max_[1] = point[1];
        }
    }

    void addRange(const Range& box)
    {
        addPoint(box.min_);
        addPoint(box.max_);
    }

    const Eigen::Vector2f& getMin() { return min_; }
    const Eigen::Vector2f& getMax() { return max_; }

private:
    Eigen::Vector2f min_;
    Eigen::Vector2f max_;
};

class LaserScan
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LaserScan() {}
    LaserScan(const PointCloud& points)
    {
        raw_points_ = points;
    }
    ~LaserScan() {}

    const PointCloud& getRawPointCloud() { return raw_points_; }
    const PointCloud& getTransformedPointCloud() { return transformed_points_; }

    void setId(int id) { id_ = id; }
    int getId() const { return id_; }
    void setPose(const Eigen::Vector3f& pose) { pose_ = pose; }
    Eigen::Vector3f getPose() const { return pose_; }
    Range getRange() const { return range_; }

    void transformPointCloud()
    {
        Range range;
        transformed_points_.clear();
        // 根据激光当前位置（x,y, yaw）构建仿射变换矩阵
        Eigen::Affine2f transform(Eigen::Translation2f(pose_[0], pose_[1]) * Eigen::Rotation2Df(pose_[2]));

        for(const Eigen::Vector2f& point : raw_points_) {
            Eigen::Vector2f transformed_point = transform * point;    // 转换激光点在以机器人坐标系下的坐标
            transformed_points_.push_back(transformed_point);
            range.addPoint(transformed_point);
        }

        range.addPoint(pose_.head<2>());

        range_ = range;
    }

private:
    PointCloud raw_points_;             //基于雷达坐标下的点云
    PointCloud transformed_points_;     //基于地图坐标系下的点云
    Eigen::Vector3f pose_;
    Range range_;
    int id_;
};

} // namespace szyh_slam

#endif // LASER_SCAN_H
