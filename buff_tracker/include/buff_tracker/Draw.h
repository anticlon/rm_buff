#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__DRAW__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__DRAW__H

#include <foxglove_msgs/msg/points_annotation.hpp>
#include <foxglove_msgs/msg/circle_annotation.hpp>
#include <foxglove_msgs/msg/text_annotation.hpp>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include "rclcpp/rclcpp.hpp"

namespace buff_tracker{

    /**
    * @brief foxglove点集标记
    * @param frame_header 时间戳
    * @param pointLists 点集
    * @param offset 偏移量
    * @param thickness 线粗
    * @param color 颜色
    * @param alpha 透明度
    */
    foxglove_msgs::msg::PointsAnnotation PointsMarker(std_msgs::msg::Header& frame_header, 
                        const std::vector<cv::Point2f>& pointLists,
                        const cv::Point2f offset,
                        float thickness, 
                        const cv::Scalar& color,
                        const float alpha);

    /**
    * @brief foxglove点集构成的ImageMarker旋转矩形
    * @param frame_header 时间戳
    * @param rotatedRect opencv旋转矩形
    * @param offset 偏移量
    * @param thickness 线粗
    * @param color 颜色
    * @param alpha 透明度
    */
    foxglove_msgs::msg::PointsAnnotation RotatedRectMarker(std_msgs::msg::Header& frame_header, 
                        const cv::RotatedRect& rotatedRect,
                        cv::Point2f offset, 
                        float thickness, 
                        const cv::Scalar& color,
                        const float alpha);

    /**
    * @brief foxglove点集构成的ImageMarker矩形
    * @param frame_header 时间戳
    * @param rect opencv矩形
    * @param offset 偏移量
    * @param thickness 线粗
    * @param color 颜色
    * @param alpha 透明度
    */
    foxglove_msgs::msg::PointsAnnotation RectMarker(std_msgs::msg::Header& frame_header,
                        const cv::Rect& rect,
                        cv::Point2f offset, 
                        float thickness, 
                        const cv::Scalar& color,
                        const float alpha = 1.0f);

    

    /**
    * @param foxglove的圆形标记
    * @param frame_header 时间戳
    * @param position 圆心坐标
    * @param diameter 直径
    * @param thickness 线粗
    * @param color 颜色
    * @param alpha 透明度
    */
    foxglove_msgs::msg::CircleAnnotation CircleMarker(std_msgs::msg::Header& frame_header,
                        const cv::Point& position,
                        float diameter,
                        float thickness, 
                        const cv::Scalar& color,
                        const float alpha = 1.0f);   

    /**
    * @brief foxglove点集构成的ImageMarker十字标
    * @param frame_header 时间戳
    * @param pt 十字中心点坐标
    * @param thickness 线粗
    * @param color 颜色
    */
    foxglove_msgs::msg::PointsAnnotation CrossMarker(std_msgs::msg::Header& frame_header,
                    const cv::Point& pt,
                    float thickness, 
                    const cv::Scalar& color);

    /**
    * @brief foxglove点集构成的ImageMarker箭头标-->
    * @param frame_header 时间戳
    * @param pt1 箭头起始点
    * @param pt2 箭头结束点
    * @param thickness 线粗
    * @param color 颜色
    */
    foxglove_msgs::msg::PointsAnnotation ArrowMarker(std_msgs::msg::Header& frame_header,
                    const cv::Point& pt1,
                    const cv::Point& pt2,
                    float thickness, 
                    const cv::Scalar& color);

    /**
    * @brief foxglove文字注释
    * @param frame_header 时间戳
    * @param pt 文字注释左上角坐标
    * @param text 文字
    * @param thickness 线粗
    * @param color 颜色
    */
    foxglove_msgs::msg::TextAnnotation TextMarker(std_msgs::msg::Header& frame_header,
                    const cv::Point& pt,
                    std::string text,
                    float thickness, 
                    const cv::Scalar& color);

    /**
     * @brief 画Opencv轮廓
     * @param src 原图像
     * @param contour 轮廓
     * @param offset 偏移量
     * @param thickness 线粗
     * @param color 颜色
    */
    void drawContour(cv::Mat& src,
                     const std::vector<cv::Point>contour,
                     const cv::Point offset, 
                     float thickness, 
                     const cv::Scalar& color);

    /**
     * @brief 按vector顺序逐点Opencv标号
     * @param src 原图像
     * @param points 点集
     * @param color 颜色(可选)
     * @param fontFace 字体(可选)
     * @param fontScale 字体比例(可选)
     * @param thickness 字体粗细(可选)
    */
    void drawNumberedPoints(cv::Mat& image, 
                            const std::vector<cv::Point2f>& points, 
                            const cv::Scalar& color = cv::Scalar(0, 0, 255), 
                            int fontFace = cv::FONT_HERSHEY_SIMPLEX,
                            double fontScale = 0.8, 
                            int thickness = 2);

}//namespace buff_tracker

#endif//IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__DRAW__H