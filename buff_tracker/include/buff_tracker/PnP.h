#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__PNP__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__PNP__H

#include "Sector.h"
#include "Values.h"
#include <array>
#include <geometry_msgs/msg/pose.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <vector>

namespace buff_tracker {
    class PnPSolver {
    public:
        PnPSolver(
                const Values *values,
                const std::array<double, 9> &camera_matrix,
                const std::vector<double> &distortion_coefficients);

        // 获取 3d 位置
        bool solvePnP(std::vector<cv::Point2f> &cameraPoints, geometry_msgs::msg::Pose &pose);

        // 计算装甲中心到图像中心的距离
        float calculateDistanceToCenter(const cv::Point2f &image_point);

        /// 转换数据类型
        static void trans(const cv::Matx33d &rmat, const cv::Mat &tvec, geometry_msgs::msg::Pose &pose);

    private:
        cv::Matx33d camera_matrix;         ///< 相机内参矩阵
        cv::Matx<double, 1, 5> dist_coeffs;///< 相机畸变矩阵
        const Values *const values;        ///< 参数

        /** @brief 3d中能量机关的6个点
        * @details 从上到下，左上, 右上，装甲板中心，左下, 右下,R中心 
        * @details 模型坐标：x 向前，y 向左，z 向上
        * @details .{1 | 2}
        * @details {---3---}
        * @details .{4 | 5}
        * @details ....|
        * @details ....|
        * @details     6(R)
        */
        std::vector<cv::Point3f> worldPoints;
    };
}//namespace buff_tracker

#endif// IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__PNP__H