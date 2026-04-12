#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__UTILITY__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__UTILITY__H

#include <Eigen/Eigen>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>

namespace buff_tracker {

    enum class Mode { SMALL, BIG };     // 模式：小符，大符
    enum class Team { BLUE , RED };
    enum class Status {
        SUCCESS,
        ARROW_FAILURE,
        ARMOR_FAILURE,
        CENTER_FAILURE
    };  // 成功，箭头检测失败，装甲板检测失败，中心R检测失败
    enum class Center_Status{
        SUCCESS,
        NO_LIGHTLINES,
        CENTER_R_NOT_FOUND
    };

    /**
     * @brief 两个二维点间距离
     * @param[in] pt1
     * @param[in] pt2
     * @return double
     */
    inline double pointPointDistance(const cv::Point2f& pt1, const cv::Point2f& pt2) noexcept {
        return std::sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
    }

    /**
     * @brief 两个三维点间距离
     * @param[in] pt1
     * @param[in] pt2
     * @return double
     */
    inline double pointPointDistance(const cv::Point3f& pt1, const cv::Point3f& pt2) noexcept {
        return std::sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) +
                        (pt1.z - pt2.z) * (pt1.z - pt2.z));
    }

    /**
     * @brief 点到直线间距离
     * @param[in] ptP
     * @param[in] ptL1
     * @param[in] ptL2
     * @return double
     */
    inline double pointLineDistance(const cv::Point2f& ptP, const cv::Point2f& ptL1,
                                    const cv::Point2f& ptL2) noexcept {
        double A = ptL2.y - ptL1.y;
        double B = ptL1.x - ptL2.x;
        double C = ptL2.x * ptL1.y - ptL2.y * ptL1.x;
        double distance = fabs(A * ptP.x + B * ptP.y + C) / sqrt(A * A + B * B);
        return distance;
    }

    /**
     * @brief 点到直线距离
     * @param[in] pt
     * @param[in] line
     * @return double
     */
    inline double pointLineDistance(const cv::Point2f& pt, const cv::Vec4f& line) {
        cv::Vec2f line_dir(line[0], line[1]);
        cv::Point2f line_pt(line[2], line[3]);
        return cv::norm((pt - line_pt).cross(line_dir));
    }

    /**
     * @brief 求解一元二次方程，返回解的数值对，第一项小于第二项
     * @param[in] a
     * @param[in] b
     * @param[in] c
     * @return std::pair<double, double>
     */
    inline std::pair<double, double> solveQuadraticEquation(double a, double b, double c) {
        std::pair<double, double> result((-b - sqrt((double)(b * b - 4 * a * c))) / (2 * a),
                                        (-b + sqrt((double)(b * b - 4 * a * c))) / (2 * a));
        return result;
    }

    /**
     * @brief 角度转弧度
     * @param[in] angle
     * @return double
     */
    inline double angle2Radian(double angle) noexcept { return angle * CV_PI / 180; }

    /**
     * @brief 弧度转角度
     * @param[in] radian
     * @return double
     */
    inline double radian2Angle(double radian) noexcept { return radian * 180 / CV_PI; }

    /**
     * @brief 判断某个值是否在一个范围内。
     * @tparam T
     * @param[in] val           判断的值
     * @param[in] lower         下限
     * @param[in] upper         上限
     * @return true
     * @return false
     */
    template <typename T>
    constexpr inline bool inRange(T val, T lower, T upper) {
        if (lower > upper) {
            std::swap(lower, upper);
        }
        return val >= lower && val <= upper;
    }

    /**
     * @brief 判断点是否在矩形内部（包括边界）
     * @param[in] point
     * @param[in] rect
     * @return true
     * @return false
     */
    inline bool inRect(const cv::Point2f& point, const cv::Rect2f& rect) {
        return point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y &&
            point.y <= rect.y + rect.height;
    }

    /**
     * @brief 计算两个 cv::Rect 之间的 CIOU。
     * @param rect1 第一个矩形。
     * @param rect2 第二个矩形。
     * @return 返回两个矩形之间的 CIOU 值。
     */
    static double calculateCIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
        // 计算两个矩形的 IoU
        cv::Rect intersection = rect1 & rect2;
        double intersectionArea = intersection.area();
        double unionArea = rect1.area() + rect2.area() - intersectionArea;
        double iou = (intersectionArea / unionArea);

        // 如果两个矩形没有重叠，IoU 为 0，CIOU 也为 0
        if (iou == 0)return 0.0;

        // 计算两个矩形的中心点
        cv::Point2f center1(rect1.x + rect1.width / 2.0, rect1.y + rect1.height / 2.0);
        cv::Point2f center2(rect2.x + rect2.width / 2.0, rect2.y + rect2.height / 2.0);

        // 计算中心点之间的距离
        double centerDistance = std::sqrt(std::pow(center1.x - center2.x, 2) + std::pow(center1.y - center2.y, 2));

        // 计算包含两个矩形的最小包围矩形的对角线长度
        double width1 = rect1.width;
        double height1 = rect1.height;
        double width2 = rect2.width;
        double height2 = rect2.height;

        double maxWidth = std::max(rect1.x + width1, rect2.x + width2) - std::min(rect1.x, rect2.x);
        double maxHeight = std::max(rect1.y + height1, rect2.y + height2) - std::min(rect1.y, rect2.y);
        double diagonalDistance = std::sqrt(std::pow(maxWidth, 2) + std::pow(maxHeight, 2));

        // 计算长宽比
        double aspectRatio1 = width1 / height1;
        double aspectRatio2 = width2 / height2;

        // CIOU 公式
        double v = 4 / (M_PI * M_PI) * std::pow(std::atan(aspectRatio1) - std::atan(aspectRatio2), 2);
        double alpha = v / (1 - iou + v);
        double ciou = iou - (std::pow(centerDistance / diagonalDistance, 2) + alpha * v);

        return ciou;
    }

    /**
     * @brief 根据图像的大小调整 roi 位置，使其不越界导致程序终止
     * @param[in] rect          待调整的 roi
     * @param[in] rows          行数
     * @param[in] cols          列数
     */
    inline void resetRoi(cv::Rect2f& rect, int rows, int cols) {
        // 调整左上角点的坐标
        rect.x = rect.x < 0 ? 0 : rect.x >= cols ? cols - 1 : rect.x;
        rect.y = rect.y < 0 ? 0 : rect.y >= rows ? rows - 1 : rect.y;
        // 调整长宽
        rect.width = rect.x + rect.width >= cols ? cols - rect.x - 1 : rect.width;
        rect.height = rect.y + rect.height >= rows ? rows - rect.y - 1 : rect.height;
        // 此时可能出现 width 或 height 小于 0 的情况，因此需要将其置为 0
        if (rect.width < 0) {
            rect.width = 0;
        }
        if (rect.height < 0) {
            rect.height = 0;
        }
    }

    inline void resetRoi(cv::Rect2f& rect, const cv::Rect2f& lastRoi) { resetRoi(rect, lastRoi.height, lastRoi.width); }

    inline cv::Point2f calculateUnitVector(const cv::Point2f& ab) {
        // 计算向量的长度（模）
        float length = std::sqrt(ab.x * ab.x + ab.y * ab.y);

        // 计算单位向量,如果长度为0，返回零向量
        return (length > 0)? cv::Point2f(ab.x / length, ab.y / length) : cv::Point2f(0, 0);
    }

    inline Team idToTeam(uint8_t robot_id){
        return (robot_id < uint8_t(50))? Team::RED : Team::BLUE;
    }

    /**
     * @brief 计算TargetShape的中心轮廓
     * @param[in] contours
     * @return 中心轮廓的外接旋转矩形
    */
    static cv::RotatedRect findCentralContour(const std::vector<std::vector<cv::Point>>& contours) {
        // 计算每个轮廓的中心点（使用 boundingRect）
        auto computeContourCenter = [](const std::vector<cv::Point>& contour) -> cv::Point {
            cv::Rect boundingBox = cv::boundingRect(contour);
            return cv::Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
        };

        // 计算所有轮廓的中心点
        std::vector<cv::Point> centers;
        for (const auto& contour : contours) {
            centers.emplace_back(computeContourCenter(contour));
        }

        // 计算所有中心点的几何中心
        cv::Point geometricCenter(0, 0);
        for (const auto& center : centers) {
            geometricCenter += center;
        }
        geometricCenter.x /= centers.size();
        geometricCenter.y /= centers.size();

        // 找到距离几何中心最近的轮廓
        double minDistance = std::numeric_limits<double>::max();
        std::vector<cv::Point> centralContour;
        for (size_t i = 0; i < contours.size(); ++i) {
            double distance = cv::norm(centers[i] - geometricCenter);
            if (distance < minDistance) {
                minDistance = distance;
                centralContour = contours[i];
            }
        }

        return cv::minAreaRect(centralContour);
    }

    /**
     * @brief 判断点 P3 在直线 P1P2 的左侧还是右侧
     * @param P1 直线起点
     * @param P2 直线终点
     * @param P3 待判断的点
     * @return 
     *    0: P3 在直线右侧
     *    1: P3 在直线左侧
     */
    inline int pointRelativeToLine(const cv::Point2f& P1, const cv::Point2f& P2, const cv::Point2f& P3) {
        // 计算叉积 (P2 - P1) × (P3 - P1)
        float cross = (P2.x - P1.x) * (P3.y - P1.y) - (P2.y - P1.y) * (P3.x - P1.x);
        return cross > 0? 1 : 0;
    }

    /**
     * @brief 用快速排斥实验 + 跨立实验快速判断线段AB与CD是否相交
     * @param A 点A坐标
     * @param B 点B坐标
     * @param C 点C坐标
     * @param D 点D坐标
     * @return  true 相交
     * @return  false 不相交
    */
    inline bool areSegmentsIntersecting(
    const cv::Point2f& A, const cv::Point2f& B,
    const cv::Point2f& C, const cv::Point2f& D){
        // 1. 快速排斥实验
        if (
            std::max(A.x, B.x) < std::min(C.x, D.x) ||
            std::max(C.x, D.x) < std::min(A.x, B.x) ||
            std::max(A.y, B.y) < std::min(C.y, D.y) ||
            std::max(C.y, D.y) < std::min(A.y, B.y)
        ) {
            return false;
        }

        // 2. 跨立实验（直接内联计算叉积）
        float cp1 = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x); // AB × AC
        float cp2 = (B.x - A.x) * (D.y - A.y) - (B.y - A.y) * (D.x - A.x); // AB × AD
        float cp3 = (D.x - C.x) * (A.y - C.y) - (D.y - C.y) * (A.x - C.x); // CD × CA
        float cp4 = (D.x - C.x) * (B.y - C.y) - (D.y - C.y) * (B.x - C.x); // CD × CB

        return (cp1 * cp2 <= 0) && (cp3 * cp4 <= 0);
    }

    inline geometry_msgs::msg::Point plane2Td(double dis,cv::Point2f pt,
                                              int height = 1080,int width = 1920,
                                              float horizon_fov = 60.0f,float vertical_fov = 30.0f){
        geometry_msgs::msg::Point point_3d;
        
        // 将FOV从度转换为弧度
        float h_fov_rad = horizon_fov * M_PI / 180.0f;
        float v_fov_rad = vertical_fov * M_PI / 180.0f;
        
        // 计算焦距（以像素为单位）
        float fx = (width / 2.0f) / tanf(h_fov_rad / 2.0f);
        float fy = (height / 2.0f) / tanf(v_fov_rad / 2.0f);
        
        // 计算图像中心坐标
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        
        // 将像素坐标转换为归一化相机坐标
        float x_norm = (pt.x - cx) / fx;
        float y_norm = (pt.y - cy) / fy;

        point_3d.x = x_norm * dis;             
        point_3d.y = y_norm * dis;     
        point_3d.z = dis;     
        
        return point_3d;
    }

    // 函数：计算旋转矩形短边与y轴的夹角（带正负）
    // 返回值：角度值，范围[-90, 90]，正表示短边逆时针偏离y轴，负表示顺时针偏离
    static double calculateShortEdgeAngleWithYAxis(const cv::RotatedRect& rect) {
        // 获取旋转矩形的角度（OpenCV返回的角度范围是[0, 90]）
        float angle = rect.angle;
        
        // 获取旋转矩形的宽高，确定哪边是短边
        float width = rect.size.width;
        float height = rect.size.height;
        
        // 确定短边对应的角度
        // OpenCV的RotatedRect角度是指长边与x轴的夹角（范围[0,90]）
        bool isWidthLonger = width > height;
        
        // 计算短边与x轴的夹角
        float shortEdgeAngle;
        if (isWidthLonger) {
            // 如果width是长边，那么短边与x轴的夹角就是angle + 90度
            shortEdgeAngle = angle + 90.0f;
        } else {
            // 如果height是长边，那么短边与x轴的夹角就是angle
            shortEdgeAngle = angle;
        }
        
        // 计算短边与y轴的夹角（转换为与y轴的夹角）
        // 我们需要的是从y轴到短边的夹角，顺时针为负，逆时针为正
        double angleWithYAxis;
        
        // 将短边与x轴的夹角转换为与y轴的夹角
        // 首先将角度限制在[0, 360)范围内
        shortEdgeAngle = fmod(shortEdgeAngle, 360.0f);
        if (shortEdgeAngle < 0) shortEdgeAngle += 360.0f;
        
        // 计算与y轴的夹角
        if (shortEdgeAngle <= 90.0f) {
            angleWithYAxis = 90.0f - shortEdgeAngle;
        } else if (shortEdgeAngle <= 180.0f) {
            angleWithYAxis = 90.0f - shortEdgeAngle;
        } else if (shortEdgeAngle <= 270.0f) {
            angleWithYAxis = -(shortEdgeAngle - 90.0f);
        } else {
            angleWithYAxis = -(shortEdgeAngle - 90.0f);
        }
        
        // 确保角度在[-90, 90]范围内
        if (angleWithYAxis > 90.0) angleWithYAxis -= 180.0;
        if (angleWithYAxis < -90.0) angleWithYAxis += 180.0;
        
        return angleWithYAxis;
    }

    static geometry_msgs::msg::Pose getOrientationTowardsOrigin(
                            const geometry_msgs::msg::Point& point,
                            double roll_offset = 0.0,
                            double pitch_offset = 0.0,
                            double yaw_offset = 0.0){
        geometry_msgs::msg::Pose pose;
        pose.position = point;

        // 计算从该点指向原点的方向向量（即Z轴负方向应对准的方向）
        Eigen::Vector3d target_dir(-point.x, -point.y, -point.z);
        target_dir.normalize();

        // 计算旋转使得Z轴负方向指向目标方向
        // 首先计算绕Z轴的旋转（yaw）
        double yaw = atan2(target_dir.y(), target_dir.x());
        
        // 然后计算绕Y轴的旋转（pitch）
        double pitch = -asin(target_dir.z());
        
        // 基础旋转（按 Z → Y 顺序，无 roll）
        Eigen::Quaterniond q_base = 
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *       // Yaw (Z)
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());      // Pitch (Y)

         // 计算固定轴偏移旋转（roll, pitch, yaw 互不影响）
        Eigen::Quaterniond q_roll(Eigen::AngleAxisd(roll_offset, Eigen::Vector3d::UnitX()));
        Eigen::Quaterniond q_pitch(Eigen::AngleAxisd(pitch_offset, Eigen::Vector3d::UnitY()));
        Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(yaw_offset, Eigen::Vector3d::UnitZ()));

        // 组合旋转（先基础旋转，再叠加 roll → pitch → yaw）
        Eigen::Quaterniond q_final = q_base * q_pitch * q_yaw * q_roll;

        // 转换为geometry_msgs::msg::Quaternion
        pose.orientation.x = q_final.x();
        pose.orientation.y = q_final.y();
        pose.orientation.z = q_final.z();
        pose.orientation.w = q_final.w();

        return pose;
    }

    
    static geometry_msgs::msg::Pose rotatePoseWithEulerAngles(
                        const geometry_msgs::msg::Pose& input_pose,
                        double roll, double pitch, double yaw){
        geometry_msgs::msg::Pose output_pose;
        
        // 保持相同的位置
        output_pose.position = input_pose.position;

        // 将输入姿态的四元数转换为Eigen格式
        Eigen::Quaterniond input_quat(
            input_pose.orientation.w,
            input_pose.orientation.x,
            input_pose.orientation.y,
            input_pose.orientation.z);

        // 创建欧拉角旋转四元数（按照Z-Y-X顺序，即yaw-pitch-roll）
        Eigen::Quaterniond rotation_quat = 
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *  // Yaw (Z轴)
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * // Pitch (Y轴)
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());   // Roll (X轴)

        // 组合旋转（先应用原始旋转，再应用欧拉角旋转）
        Eigen::Quaterniond output_quat = input_quat * rotation_quat;

        // 转换为geometry_msgs格式
        output_pose.orientation.x = output_quat.x();
        output_pose.orientation.y = output_quat.y();
        output_pose.orientation.z = output_quat.z();
        output_pose.orientation.w = output_quat.w();

        return output_pose;
    }

    static geometry_msgs::msg::Point moveAlongPoseDirection(
            const geometry_msgs::msg::Pose& pose,
            float distance){
        geometry_msgs::msg::Point new_point;

        // 将姿态的四元数转换为Eigen格式
        Eigen::Quaternionf quat(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);

        // 获取姿态的前向方向（假设为X轴正方向）
        Eigen::Vector3f forward_dir = quat * Eigen::Vector3f::UnitX();

        // 计算新位置
        new_point.x = pose.position.x + forward_dir.x() * distance;
        new_point.y = pose.position.y + forward_dir.y() * distance;
        new_point.z = pose.position.z + forward_dir.z() * distance;

        return new_point;
    }

    inline cv::Rect RoiX2(cv::Rect rgb){
        return cv::Rect(rgb.x*2,rgb.y*2,rgb.width*2,rgb.height*2);
    }

    inline cv::Rect RoiD2(cv::Rect bayer){
        return cv::Rect(bayer.x/2,bayer.y/2,bayer.width/2,bayer.height/2);
    }

    inline cv::RotatedRect RRX2(cv::RotatedRect rotatedRect){
        return cv::RotatedRect(rotatedRect.center*2,{rotatedRect.size.width*2,rotatedRect.size.height*2},rotatedRect.angle);
    }

    // 计算旋转矩形的方向向量（与长边平行）
    // rotatedRect: 输入的旋转矩形
    // referencePoint: 参考点，用于确定方向向量的朝向
    // 返回值: 单位方向向量（cv::Point2f）
    static cv::Point2f calculateDirectionVector(const cv::RotatedRect& rotatedRect, const cv::Point2f& referencePoint) {
        // 获取旋转矩形的四个顶点
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);
        
        // 确定长边和短边
        float edge1Length = cv::norm(vertices[1] - vertices[0]);
        float edge2Length = cv::norm(vertices[2] - vertices[1]);
        
        // 选择长边作为方向基准
        cv::Point2f longEdgeVector;
        if (edge1Length > edge2Length) {
            longEdgeVector = vertices[1] - vertices[0];
        } else {
            longEdgeVector = vertices[2] - vertices[1];
        }
        
        // 计算长边的两个可能方向
        cv::Point2f direction1 = longEdgeVector / cv::norm(longEdgeVector);
        cv::Point2f direction2 = -direction1;
        
        // 计算旋转矩形中心
        cv::Point2f center = rotatedRect.center;
        
        // 计算参考点到两个可能方向向量的投影距离
        cv::Point2f refVec = referencePoint - center;
        float proj1 = direction1.dot(refVec);
        float proj2 = direction2.dot(refVec);
        
        // 选择与参考点方向更一致的方向
        return (proj1 > proj2) ? direction1 : direction2;
    }

}  // namespace buff_tracker

#endif// IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__UTILITY__H
