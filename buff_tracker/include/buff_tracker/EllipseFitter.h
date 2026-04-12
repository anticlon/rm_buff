#ifndef ELLIPSE_FITTER_H
#define ELLIPSE_FITTER_H

#include <vector>
#include <deque>
#include <memory>
#include "Values.h"
#include <opencv2/opencv.hpp>

namespace buff_tracker{
    class EllipseFitter {
public:
    struct Parameters {
        size_t max_points;                  // 最大存储点数
        float inlier_threshold;             // 内点距离阈值(像素)
        int min_samples;                    // 最小拟合样本数
        float confidence;                   // RANSAC置信度
        int max_iterations;                 // RANSAC最大迭代次数
        float min_inlier_ratio;             // 最小内点比例
        float min_point_distance;           // 最小点间距阈值
        size_t max_mapping_history;         // 最大历史记录数
    };

    // 拟合质量评估
    struct FitQuality {
        double mean_distance;               // 所有内点到椭圆边缘的平均几何距离
        double max_distance;                // 所有内点中到椭圆边缘的最大距离
        double inlier_ratio;                // 内点占全部点集的比例
        double confidence;                  // 模型可靠性的概率化估计(基于RANSAC算法)
        int iterations;                     // RANSAC算法实际运行的迭代次数
    };

    // RANSAC状态
    struct RansacState {
        int iterations = 0;
        int best_inliers = 0;
        cv::RotatedRect best_model;
    };

    // 新增映射结果结构体
    struct MappingResult {
        cv::Point2f input_point;     // 原始输入点
        cv::Point2f map_point;       // 映射点
        bool valid = false;          // 是否有效
    };

    // 初始化
    explicit EllipseFitter(const Values *values);
    
    // 点集操作
    void addPoint(const cv::Point2f& point);                    // 直接加入点
    bool addPointWithDistanceCheck(const cv::Point2f& point);   // 带距离监测的加入点
    void clearPoints();                                         // 清空点集
    size_t getCurrentPointsCount() const;                       // 获取目前点数
    
    // 椭圆拟合
    bool fitIncrementalRANSAC();
    bool fitStandard();
    
    // 结果获取
    cv::RotatedRect getFittedEllipse() const;               // 获取拟合椭圆
    const std::vector<cv::Point2f>& getInliers() const;     // 获取内点集合
    const std::vector<cv::Point2f>& getOutliers() const;    // 获取离群点集合
    const float& getCircumscribedRadius() const;            // 获取外切圆半径
    const std::string& getLastError() const;                // 获取错误信息
    bool isFit() const;                                     // 是否完成拟合
    cv::Point2f getCenter(cv::Point2f default_val = cv::Point2f(-1,-1)) const;  // 获取拟合椭圆中心

    
    FitQuality getFitQuality() const;                       // 获取拟合椭圆质量信息
    
    // 可视化
    void visualize(cv::Mat& image, 
                 const cv::Point2f& offset = cv::Point2f(0, 0),
                 float scale = 1.0f) const;
    
    // 获取/设置最小点间距
    float getMinPointDistance() const;                      // 获取最小点间距
    void setMinPointDistance(float distance);               // 设置最小点间距

    // 点映射
    cv::Point2f mapPoint(const cv::Point2f& point);

private:
    Parameters params_;                         // 参数数据
    std::deque<cv::Point2f> points_;            // 数据点集合
    cv::RotatedRect fitted_ellipse_;            // 椭圆拟合结果
    std::vector<cv::Point2f> inliers_,outliers_;// 内点集合和离群点集合
    std::string last_error_;                    // 最后错误信息
    FitQuality fit_quality_;                    // 拟合椭圆质量信息
    std::deque<MappingResult> mapping_history_; // 保存映射结果
    float CircumscribedRadius;                  // 外切圆半径信息 
    float h_ratio;                              // 长轴高度比
    
    std::unique_ptr<RansacState> ransac_state_; // RANSAC状态
    
    bool basicEllipseFitting(const std::vector<cv::Point2f>& points, cv::RotatedRect& ellipse);
    void evaluateFitQuality();
    static double pointToEllipseDistance(const cv::Point2f& pt, const cv::RotatedRect& ellipse);
    void updateInliersOutliers();
    bool isPointFarEnough(const cv::Point2f& point) const;
};
}// namespace buff_tracker

#endif // ELLIPSE_FITTER_H