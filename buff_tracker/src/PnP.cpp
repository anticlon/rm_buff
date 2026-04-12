#include "buff_tracker/PnP.h"
#include <cmath>
#include <rclcpp/logging.hpp>

namespace buff_tracker{
    PnPSolver::PnPSolver(
            const Values *values,
            const std::array<double, 9> &camera_matrix,
            const std::vector<double> &dist_coeffs)
        : values(values),
          camera_matrix(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
          dist_coeffs(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone()) {

        //单位: 米
        // 从上到下，左上, 右上，装甲板中心，左下, 右下,R中心 
        // 模型坐标：x 向前，y 向左，z 向上
        // 以中心R为原点
        worldPoints.emplace_back(0.0, -0.1246, 0.8246);
        worldPoints.emplace_back(0.0,  0.1246, 0.8246);
        worldPoints.emplace_back(0.0,  0.0,    0.7   );
        worldPoints.emplace_back(0.0, -0.1246, 0.5754);
        worldPoints.emplace_back(0.0,  0.1246, 0.5754);
        worldPoints.emplace_back(0.0,  0.0,    0.0   );
    }

    bool PnPSolver::solvePnP(std::vector<cv::Point2f> &cameraPoints, geometry_msgs::msg::Pose &pose) {
        cv::Mat rvec, tvec;

        cv::solvePnP(worldPoints, cameraPoints, camera_matrix, dist_coeffs, rvec, tvec, false,
                    cv::SOLVEPNP_IPPE);

        cv::Matx33d R;
        cv::Rodrigues(rvec, R);

        trans(R, tvec, pose);

        return true;
    }

    float PnPSolver::calculateDistanceToCenter(const cv::Point2f &image_point) {
        float cx = camera_matrix(0, 2);
        float cy = camera_matrix(1, 2);
        return cv::norm(image_point - cv::Point2f(cx, cy));
    }


    //将 PnP 解算的旋转矩阵和平移向量，转换为 ROS 的geometry_msgs::msg::Pose位姿消息
    void PnPSolver::trans(const cv::Matx33d &rmat, const cv::Mat &tvec, geometry_msgs::msg::Pose &pose) {
        // Fill armor_msg with pose
        pose.position.x = tvec.at<double>(0);
        pose.position.y = tvec.at<double>(1);
        pose.position.z = tvec.at<double>(2);
        // rotation matrix to quaternion
        tf2::Matrix3x3 tf2_rotation_matrix(
                rmat(0, 0), rmat(0, 1), rmat(0, 2),
                rmat(1, 0), rmat(1, 1), rmat(1, 2),
                rmat(2, 0), rmat(2, 1), rmat(2, 2));
        tf2::Quaternion tf2_quaternion;//= tf2_quaternion.getIdentity();
        tf2_rotation_matrix.getRotation(tf2_quaternion);
        pose.orientation.x = tf2_quaternion.x();
        pose.orientation.y = tf2_quaternion.y();
        pose.orientation.z = tf2_quaternion.z();
        pose.orientation.w = tf2_quaternion.w();
    }
}// namespace rm_armor_finder