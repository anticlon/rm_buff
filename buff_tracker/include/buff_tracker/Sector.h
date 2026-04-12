#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__SECTOR__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__SECTOR__H

#include <opencv2/opencv.hpp>
#include "buff_tracker/Utility.h"

namespace buff_tracker{
    
    /**
     * @brief 流水灯的单个灯条
     */
    struct Lightline {
        Lightline() = default;
        Lightline(const std::vector<cv::Point>& contour, const cv::Rect2f& globalRoi,
                const cv::Rect2f& localroi);
        std::vector<cv::Point> m_contour;  // 轮廓点集
        double m_contourArea;              // 轮廓面积
        double m_area;                     // 外接旋转矩形面积
        cv::RotatedRect m_rotatedRect;     // 外接旋转矩形
        cv::Point2f m_tl;                  // 左上角点
        cv::Point2f m_tr;                  // 右上角点
        cv::Point2f m_bl;                  // 左下角点
        cv::Point2f m_br;                  // 右下角点
        cv::Point2f m_center;              // 中心点
        double m_length;                   // 长度
        double m_width;                    // 宽度
        double m_x;                        // 中心点 x 坐标
        double m_y;                        // 中心点 y 坐标
        double m_angle;                    // 旋转矩形角度
        double m_aspectRatio;              // 旋转矩形长宽比 (>1)
    };

    /**
     * @brief 击打十字标图案
     */
    struct TargetShape{
        TargetShape() = default;
        TargetShape(const std::vector<std::vector<cv::Point>>& contours,std::vector<cv::Vec4i>& hierarchy,int contourIndex,const cv::Rect2f& globalRoi,
                const cv::Rect2f& localroi);
        std::vector<cv::Point> m_contour;                       // 轮廓点集
        std::vector<std::vector<cv::Point>> m_childrenContours; // 子轮廓集 
        int m_children;                                         // 子层级数量
        double m_area;                                          // 外接旋转矩形面积
        double m_counterArea;                                   // 轮廓面积
        cv::RotatedRect m_rotatedRect;                          // 外接旋转矩形
        cv::Point2f m_tl;                                       // 左上角点
        cv::Point2f m_tr;                                       // 右上角点
        cv::Point2f m_bl;                                       // 左下角点
        cv::Point2f m_br;                                       // 右下角点
        cv::Point2f m_center;                                   // 中心点
        double m_length;                                        // 长度
        double m_width;                                         // 宽度
        double m_x;                                             // 中心点 x 坐标
        double m_y;                                             // 中心点 y 坐标
        double m_angle;                                         // 旋转矩形角度
        double m_aspectRatio;                                   // 旋转矩形长宽比 (>1)
        cv::Point2f m_roi;                                      // 全局ROI
    };

    /**
     * @brief 装甲板
     */
    struct Armor {
        Armor() = default;
        void set(const TargetShape& tar);
        void setCornerPoints(const std::vector<cv::Point2f>& points);
        TargetShape m_targetshape;
        cv::RotatedRect m_centerRotatedRect;
        cv::Point2f m_center;  // 装甲板中心点
        double m_x;            // 装甲板中心 x 坐标
        double m_y;            // 装甲板中心 y 坐标
    };

    /**
     * @brief 中心 R
     */
    struct CenterR {
        CenterR() = default;
        void set(const Lightline& contour);
        Lightline m_lightline;    // 中心 R 灯条
        cv::Point2f m_center;     // 中心 R 点
        cv::Rect m_boundingRect;  // 中心 R 最小正矩形
        double m_x;               // 中心 R x 坐标
        double m_y;               // 中心 R y 坐标
    };

    /**
     * @brief 箭头
     */
    struct Arrow {
        Arrow() = default;
        void set(const std::vector<Lightline>& points, const cv::Point2f& roi);
        std::vector<cv::Point> m_contour;  // 轮廓点集
        cv::RotatedRect m_rotatedRect;     // 外接旋转矩形
        double m_length;                   // 长度
        double m_width;                    // 宽度
        cv::Point2f m_center;              // 中心点
        double m_angle;                    // 角度
        double m_aspectRatio;              // 长宽比
        double m_area;                     // 面积
        double m_fillRatio;                // 填充比例
    };

}// namespace buff_tracker

#endif// IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__SECTOR__H