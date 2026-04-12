#include "buff_tracker/EllipseFitter.h"
#include <random>
#include <cmath>

namespace buff_tracker{
    EllipseFitter::EllipseFitter(const Values *values){
        // 参数设定
        params_.confidence = values->RANSAC_CONFIDENCE;
        params_.inlier_threshold = values->INLIER_THRESHOLD;
        params_.max_iterations = values->MAX_ITERATIONS;
        params_.max_mapping_history = values->MAX_MAPPING_HISTORY;
        params_.max_points = values->MAX_POINTS;
        params_.min_inlier_ratio = values->MIN_INLIER_RATIO;
        params_.min_point_distance = values->MIN_POINT_DISTENCE;
        params_.min_samples = values->MIN_SAMPLES;

        // 变量初始化
        CircumscribedRadius = 0;
    }

    cv::Point2f EllipseFitter::getCenter(cv::Point2f default_val) const {
        return fitted_ellipse_.size.empty() ? default_val : fitted_ellipse_.center;
    }

    bool EllipseFitter::isFit()const{
        return !(fitted_ellipse_.size.empty());
    }

    void EllipseFitter::addPoint(const cv::Point2f& point) {
        if (params_.max_points > 0 && points_.size() >= params_.max_points) {
            points_.pop_front();
        }
        points_.emplace_back(point);
        
        // 增量式RANSAC更新
        if (points_.size() >= static_cast<size_t>(params_.min_samples)) {
            fitIncrementalRANSAC();
        }
    }

    void EllipseFitter::clearPoints() {
        points_.clear();
        inliers_.clear();
        outliers_.clear();
        ransac_state_.reset();
    }

    size_t EllipseFitter::getCurrentPointsCount() const {
        return points_.size();
    }

    bool EllipseFitter::fitIncrementalRANSAC() {
        if (points_.size() < static_cast<size_t>(params_.min_samples)) {
            last_error_ = "Not enough points for fitting";
            return false;
        }

        // 初始化RANSAC状态
        if (!ransac_state_) {
            ransac_state_ = std::make_unique<RansacState>();
        }

        // 转换点集为vector
        std::vector<cv::Point2f> all_points(points_.begin(), points_.end());
        
        // RANSAC参数
        int N = params_.max_iterations;
        const float t = params_.inlier_threshold;
        const int k = params_.min_samples;
        
        // 随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, static_cast<int>(all_points.size()) - 1);
        
        int best_inliers = ransac_state_->best_inliers;
        cv::RotatedRect best_model = ransac_state_->best_model;
        int iter = 0;
        
        // 增量式RANSAC迭代
        for (; iter < N; ++iter) {
            // 随机选择k个点
            std::vector<cv::Point2f> sample;
            for (int i = 0; i < k; ++i) {
                sample.emplace_back(all_points[dis(gen)]);
            }
            
            // 拟合椭圆
            cv::RotatedRect ellipse;
            if (!basicEllipseFitting(sample, ellipse)) {
                continue;
            }
            
            // 统计内点
            int inliers = 0;
            for (const auto& pt : all_points) {
                if (pointToEllipseDistance(pt, ellipse) <= t) {
                    inliers++;
                }
            }
            
            // 更新最佳模型
            if (inliers > best_inliers) {
                best_inliers = inliers;
                best_model = ellipse;
                
                // 动态调整迭代次数
                double inlier_ratio = static_cast<double>(inliers) / all_points.size();
                if (inlier_ratio > params_.min_inlier_ratio) {
                    double log_prob = log(1.0 - params_.confidence);
                    double log_inlier = log(1.0 - pow(inlier_ratio, k));
                    N = static_cast<int>(log_prob / log_inlier) + 1;
                    N = std::min(N, params_.max_iterations);
                }
            }
        }
        
        // 更新状态
        ransac_state_->iterations += iter;
        ransac_state_->best_inliers = best_inliers;
        ransac_state_->best_model = best_model;
        fitted_ellipse_ = best_model;
        
        // 更新内点和离群点
        updateInliersOutliers();
        evaluateFitQuality();
        
        // 检查拟合是否有效
        double inlier_ratio = static_cast<double>(best_inliers) / all_points.size();
        if (inlier_ratio < params_.min_inlier_ratio) {
            last_error_ = "Insufficient inliers: " + std::to_string(inlier_ratio);
            return false;
        }
        
        return true;
    }

    bool EllipseFitter::fitStandard() {
        if (points_.size() < static_cast<size_t>(params_.min_samples)) {
            last_error_ = "Not enough points for fitting";
            return false;
        }
        
        std::vector<cv::Point2f> points_vec(points_.begin(), points_.end());
        try {
            fitted_ellipse_ = cv::fitEllipse(points_vec);
            updateInliersOutliers();
            evaluateFitQuality();
            return true;
        } catch (const cv::Exception& e) {
            last_error_ = e.what();
            return false;
        }
    }

    const float& EllipseFitter::getCircumscribedRadius() const{
        return CircumscribedRadius;
    }

    cv::RotatedRect EllipseFitter::getFittedEllipse() const {
        return fitted_ellipse_;
    }

    const std::vector<cv::Point2f>& EllipseFitter::getInliers() const {
        return inliers_;
    }

    const std::vector<cv::Point2f>& EllipseFitter::getOutliers() const {
        return outliers_;
    }

    const std::string& EllipseFitter::getLastError() const {
        return last_error_;
    }

    EllipseFitter::FitQuality EllipseFitter::getFitQuality() const {
        return fit_quality_;
    }

    void EllipseFitter::visualize(cv::Mat& image, 
                            const cv::Point2f& offset,
                            float scale) const {
        // 绘制所有点(内点绿色，离群点红色)
        for (const auto& pt : points_) {
            cv::Point2f scaled_pt(pt.x * scale + offset.x, pt.y * scale + offset.y);
            bool is_inlier = std::find(inliers_.begin(), inliers_.end(), pt) != inliers_.end();
            cv::Scalar color = is_inlier ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 200);
            cv::circle(image, scaled_pt, 2, color, cv::FILLED);
        }

        // 如果拟合成功，绘制椭圆和相关元素
        if (!fitted_ellipse_.size.empty()) {
            // 计算缩放后的椭圆参数
            cv::RotatedRect scaled_ellipse;
            scaled_ellipse.center.x = fitted_ellipse_.center.x * scale + offset.x;
            scaled_ellipse.center.y = fitted_ellipse_.center.y * scale + offset.y;
            scaled_ellipse.size.width = fitted_ellipse_.size.width * scale;
            scaled_ellipse.size.height = fitted_ellipse_.size.height * scale;
            scaled_ellipse.angle = fitted_ellipse_.angle;

            // 绘制拟合的椭圆(绿色)
            cv::ellipse(image, scaled_ellipse, cv::Scalar(0, 255, 0), 2);
            
            // 绘制椭圆中心(蓝色)
            cv::circle(image, scaled_ellipse.center, 3, cv::Scalar(255, 0, 0), cv::FILLED);

            // 获取并绘制最小外切圆(粉色)
            cv::Point2f circle_center = scaled_ellipse.center;
            float radius = std::max(scaled_ellipse.size.width, scaled_ellipse.size.height) / 2.0f;
            cv::circle(image, circle_center, radius, cv::Scalar(255, 200, 255), 1);

            // 显示拟合信息
            std::string info = "Pts: " + std::to_string(points_.size()) + "/" + std::to_string(params_.max_points);
            info += ("In:" + std::to_string(int(fit_quality_.inlier_ratio * 100)) + "%");
                
            // 绘制带背景的文字
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(image, 
                        cv::Point(offset.x + 5, offset.y + 5),
                        cv::Point(offset.x + 10 + text_size.width, offset.y + 25 + text_size.height),
                        cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(image, info, cv::Point(offset.x + 10, offset.y + 25), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // 绘制映射结果
        for (const auto& mapping : mapping_history_) {
            if (!mapping.valid) continue;

            // 缩放和偏移所有点
            auto transformPoint = [&](const cv::Point2f& pt) {
                return cv::Point2f(pt.x * scale + offset.x, 
                                pt.y * scale + offset.y);
            };

            const cv::Point2f input = transformPoint(mapping.input_point);
            const cv::Point2f map_ = transformPoint(mapping.map_point);

            // 绘制原始点（红色）
            cv::circle(image, input, 5, cv::Scalar(0, 0, 255), cv::FILLED);
            
            // 绘制外切圆映射点（绿色）
            cv::circle(image, map_, 5, cv::Scalar(0, 255, 0), 2);
            
            // 绘制连接线
            cv::line(image, input, map_, cv::Scalar(200, 200, 200), 1);
            
            // 添加文字标注
            cv::putText(image, "Input", input + cv::Point2f(5, -5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255), 1);
            cv::putText(image, "Projection", map_ + cv::Point2f(5, -5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,0), 1);
        }
    }

    bool EllipseFitter::basicEllipseFitting(const std::vector<cv::Point2f>& points, cv::RotatedRect& ellipse) {
        if (points.size() < static_cast<size_t>(params_.min_samples)) {
            return false;
        }
        try {
            ellipse = cv::fitEllipse(points);
            return true;
        } catch (...) {
            return false;
        }
    }

    void EllipseFitter::evaluateFitQuality() {
        if (inliers_.empty()) {
            fit_quality_ = FitQuality{0, 0, 0, 0, 0};
            return;
        }

        // 计算距离统计
        double sum_dist = 0;
        double max_dist = 0;
        for (const auto& pt : inliers_) {
            double dist = pointToEllipseDistance(pt, fitted_ellipse_);
            sum_dist += dist;
            if (dist > max_dist) max_dist = dist;
        }

        fit_quality_.mean_distance = sum_dist / inliers_.size();
        fit_quality_.max_distance = max_dist;
        fit_quality_.inlier_ratio = static_cast<double>(inliers_.size()) / points_.size();
        fit_quality_.iterations = ransac_state_ ? ransac_state_->iterations : 0;
        
        // 计算综合置信度
        double confidence = 0.4 * (1.0 - fit_quality_.mean_distance / params_.inlier_threshold);
        confidence += 0.4 * fit_quality_.inlier_ratio;
        confidence += 0.2 * (1.0 - 1.0 / (1.0 + exp(-(static_cast<int>(points_.size()) - 20) / 10.0)));
        fit_quality_.confidence = std::clamp(confidence, 0.0, 1.0);
    }

    double EllipseFitter::pointToEllipseDistance(const cv::Point2f& pt, const cv::RotatedRect& ellipse) {
        cv::Point2f translated = pt - ellipse.center;
        double angle = -ellipse.angle * CV_PI / 180.0;
        double cos_angle = cos(angle);
        double sin_angle = sin(angle);
        
        double x = translated.x * cos_angle - translated.y * sin_angle;
        double y = translated.x * sin_angle + translated.y * cos_angle;
        
        double a = ellipse.size.width / 2.0;
        double b = ellipse.size.height / 2.0;
        return abs((x*x)/(a*a) + (y*y)/(b*b) - 1.0) * std::min(a, b);
    }

    void EllipseFitter::updateInliersOutliers() {
        inliers_.clear();
        outliers_.clear();
        
        if (fitted_ellipse_.size.empty()) return;
        
        for (const auto& pt : points_) {
            if (pointToEllipseDistance(pt, fitted_ellipse_) <= params_.inlier_threshold) {
                inliers_.emplace_back(pt);
            } else {
                outliers_.emplace_back(pt);
            }
        }
    }

    bool EllipseFitter::addPointWithDistanceCheck(const cv::Point2f& point) {
        if (points_.empty() || isPointFarEnough(point)) {
            addPoint(point);
            return true;
        }
        return false;
    }

    float EllipseFitter::getMinPointDistance() const {
        return params_.min_point_distance;
    }

    void EllipseFitter::setMinPointDistance(float distance) {
        params_.min_point_distance = std::max(0.0f, distance);
    }

    bool EllipseFitter::isPointFarEnough(const cv::Point2f& point) const {
        if (points_.empty()) {
            return true;
        }
        
        const cv::Point2f& last_point = points_.back();
        float dx = point.x - last_point.x;
        float dy = point.y - last_point.y;
        float distance = std::sqrt(dx*dx + dy*dy);
        
        return distance >= params_.min_point_distance;
    }

    cv::Point2f EllipseFitter::mapPoint(const cv::Point2f& point){
        MappingResult result;
        result.input_point = point;
        result.valid = false;
        
        if (fitted_ellipse_.size.empty()) {
            return cv::Point2f(0,0);
            throw std::runtime_error("No ellipse fitted yet");
        }

        const cv::Point2f center = fitted_ellipse_.center;
        const float theta = -fitted_ellipse_.angle * CV_PI / 180.0f;  // 转换旋转角度到弧度

        // 将点转换到以椭圆中心为原点的坐标系
        cv::Point2f translated = point - center;

        // 反向旋转消除椭圆方向影响
        float x_rot = translated.x * cos(theta) - translated.y * sin(theta);
        float y_rot = translated.x * sin(theta) + translated.y * cos(theta);

        // 计算当前点在标准椭圆坐标系中的归一化坐标
        float a = fitted_ellipse_.size.width / 2.0f;
        float b = fitted_ellipse_.size.height / 2.0f;
        this->CircumscribedRadius = std::max(a, b);
        float Xscale = CircumscribedRadius / a;  // 统一缩放比例
        float Yscale = CircumscribedRadius / b;  // 统一缩放比例
        this->h_ratio = Yscale;

        // 将点映射到外接圆（坐标系仍以椭圆中心为原点）
        cv::Point2f mapped(x_rot * Xscale, y_rot * Yscale);

        // 旋转回原始方向并平移回原坐标系
        cv::Point2f rotated_back(
            mapped.x * cos(-theta) - mapped.y * sin(-theta),
            mapped.x * sin(-theta) + mapped.y * cos(-theta)
        );

        // 存储计算结果
        result.map_point = rotated_back + center;
        result.valid = true;

        // 维护历史记录
        mapping_history_.emplace_back(result);
        if (mapping_history_.size() > params_.max_mapping_history) {
            mapping_history_.pop_front();
        }

        return rotated_back + center;
    }


}//namespace buff_tracker