#include "buff_tracker/Sector.h"

namespace buff_tracker{
    /**
     * @brief Construct a new TargetShape:: TargetShape object
     * @param[in] contour       轮廓点集
     * @param[in] roi           roi 用来设置正确的中心及角点
     */
    TargetShape::TargetShape(const std::vector<std::vector<cv::Point>>& contours,std::vector<cv::Vec4i>& hierarchy,int contourIndex,
                const cv::Rect2f& globalRoi,const cv::Rect2f& localRoi)
        : m_contour(contours[contourIndex]), m_counterArea(cv::contourArea(contours[contourIndex])),m_rotatedRect(cv::minAreaRect(contours[contourIndex])) {
        // 长的为 length，短的为 width
        m_width = m_rotatedRect.size.width, m_length = m_rotatedRect.size.height;
        if (m_width > m_length) {
            std::swap(m_width, m_length);
        }
        // he ge zi lun kuo
        int childCount = 0;
        int currentChildIndex = hierarchy[contourIndex][2];
        // 遍历子轮廓，直到没有下一个子轮廓为止
        while (currentChildIndex!= -1) {
            auto temp = contours[currentChildIndex];
            auto temp_contourArea = cv::contourArea(temp);
            auto c_rotatedRect = cv::minAreaRect(temp);
            float c_length = c_rotatedRect.size.height;
            float c_width = c_rotatedRect.size.width;
            if (c_length < c_width)std::swap(c_length , c_width);
            float c_aspectRatio = c_length / c_width;
            if (temp_contourArea > 10 && c_aspectRatio > 1.3){
                childCount++;
                m_childrenContours.emplace_back(temp);
            }
            // 获取下一个子轮廓的索引
            currentChildIndex = hierarchy[currentChildIndex][0];
        }
        m_children = childCount;
        m_aspectRatio = m_length / m_width;
        m_center = m_rotatedRect.center;
        m_angle = m_rotatedRect.angle;
        m_area = m_rotatedRect.size.width * m_rotatedRect.size.height;
        std::array<cv::Point2f, 4> points;
        m_rotatedRect.points(points.begin());
        /**
         * OpenCV 中 RotatedRect::points() 角点顺序为顺时针，p[0]
         * 为纵坐标最大的点。若有多个纵坐标最大，则取其中横坐标最大的点。 p[0] 到 p[3] 的边为 width，其邻边为
         * height。 根据上述关系可以确立四个角点位置。如果是装甲板灯条，则其还需要结合中心 R 来得到中心 R
         * 参照下的角点位置。
         */
        if (m_rotatedRect.size.width > m_rotatedRect.size.height) {
            m_tl = points[1];
            m_tr = points[2];
            m_bl = points[0];
            m_br = points[3];
        } else {
            m_tl = points[0];
            m_tr = points[1];
            m_bl = points[3];
            m_br = points[2];
        }
        // 得到相对原图的角点和中心位置
        m_roi = localRoi.tl() + globalRoi.tl();
        m_tl += m_roi;
        m_tr += m_roi;
        m_bl += m_roi;
        m_br += m_roi;
        m_center += m_roi;
        m_x = m_center.x, m_y = m_center.y;
    }

    /**
     * @brief Construct a new Lightline:: Lightline object
     * @param[in] contour       轮廓点集
     * @param[in] roi           roi 用来设置正确的中心及角点
     */
    Lightline::Lightline(const std::vector<cv::Point>& contour, const cv::Rect2f& localRoi,
                        const cv::Rect2f& globalRoi)
        : m_contour(contour), m_contourArea(cv::contourArea(contour)), m_rotatedRect(cv::minAreaRect(contour)) {
        // 长的为 length，短的为 width
        m_width = m_rotatedRect.size.width, m_length = m_rotatedRect.size.height;
        if (m_width > m_length) {
            std::swap(m_width, m_length);
        }
        m_aspectRatio = m_length / m_width;
        m_center = m_rotatedRect.center;
        m_angle = m_rotatedRect.angle;
        m_area = m_rotatedRect.size.width * m_rotatedRect.size.height;
        std::array<cv::Point2f, 4> points;
        m_rotatedRect.points(points.begin());
        /**
         * OpenCV 中 RotatedRect::points() 角点顺序为顺时针，p[0]
         * 为纵坐标最大的点。若有多个纵坐标最大，则取其中横坐标最大的点。 p[0] 到 p[3] 的边为 width，其邻边为
         * height。 根据上述关系可以确立四个角点位置。如果是装甲板灯条，则其还需要结合中心 R 来得到中心 R
         * 参照下的角点位置。
         */
        if (m_rotatedRect.size.width > m_rotatedRect.size.height) {
            m_tl = points[1];
            m_tr = points[2];
            m_bl = points[0];
            m_br = points[3];
        } else {
            m_tl = points[0];
            m_tr = points[1];
            m_bl = points[3];
            m_br = points[2];
        }
        // 得到相对原图的角点和中心位置
        m_tl += localRoi.tl() + globalRoi.tl();
        m_tr += localRoi.tl() + globalRoi.tl();
        m_bl += localRoi.tl() + globalRoi.tl();
        m_br += localRoi.tl() + globalRoi.tl();
        m_center += localRoi.tl() + globalRoi.tl();
        m_x = m_center.x, m_y = m_center.y;
    }

    /**
     * @brief 设置中心 R
     * @param[in] lightline
     */
    void CenterR::set(const Lightline& lightline) {
        m_lightline = lightline;
        m_boundingRect = cv::boundingRect(lightline.m_contour);
        // 由于灯条角点和中心点已经设置过 roi，因此这里不需要重新设置
        m_center = lightline.m_center;
        m_x = m_center.x, m_y = m_center.y;
        return;
    }
    
    /**
     * @brief 设置箭头
     * @param[in] points        点集
     * @param[in] roi
     */
    void Arrow::set(const std::vector<Lightline>& lightlines, const cv::Point2f& roi) {
        std::vector<cv::Point2f> arrowPoints;
        double fillArea = 0.0;
        double pointLineThresh = 0.0;
        std::for_each(lightlines.begin(), lightlines.end(), [&](const Lightline& l) {
            arrowPoints.insert(arrowPoints.end(), l.m_contour.begin(), l.m_contour.end());
            fillArea += l.m_contourArea;
            pointLineThresh += l.m_length / lightlines.size();
        });
        // 滤除距离较大的点
        m_contour.clear();
        cv::Vec4f line;
        cv::fitLine(arrowPoints, line, cv::DIST_L2, 0, 0.01, 0.01);
        for (const auto& point : arrowPoints) {
            if (pointLineDistance(point, line) < pointLineThresh) {
                m_contour.emplace_back(point);
            }
        }
        // 设置成员变量
        m_rotatedRect = cv::minAreaRect(m_contour);
        m_center = m_rotatedRect.center + roi;
        m_length = m_rotatedRect.size.height;
        m_width = m_rotatedRect.size.width;
        // RotatedRect::angle 范围为 -90~0. 这里根据长宽长度关系，将角度扩展到 -90~90
        if (m_length < m_width) {
            m_angle = m_rotatedRect.angle;
            // 长的为 length
            std::swap(m_length, m_width);
        } else {
            m_angle = m_rotatedRect.angle + 90;
        }
        m_aspectRatio = m_length / m_width;
        m_area = m_length * m_width;
        m_fillRatio = fillArea / m_area;
        return;
    }

        /**
     * @brief 设置装甲板参数
     * @param[in] l1
     * @param[in] l2
     */
    void Armor::set(const TargetShape& tar) {
        //float x_sum = 0.0;
        //float y_sum = 0.0;
        m_targetshape = tar;
        //m_centerRotatedRect = findCentralContour(tar.m_childrenContours);
        /*
        for(const auto& contour : tar.m_childrenContours){
            auto temp = cv::minAreaRect(contour);
            x_sum += temp.center.x;
            y_sum += temp.center.y;
        }
        */
        //m_center = cv::Point2f(x_sum/tar.m_childrenContours.size(),y_sum/tar.m_childrenContours.size());
        m_center = m_targetshape.m_center/*+ tar.m_roi*/;
        m_x = m_center.x, m_y = m_center.y;
        return;
    }

}//namespace buff_tracker