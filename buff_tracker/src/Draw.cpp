#include "buff_tracker/Draw.h"

namespace buff_tracker{

    foxglove_msgs::msg::PointsAnnotation PointsMarker(std_msgs::msg::Header& frame_header, 
                const std::vector<cv::Point2f>& pointLists,
                const cv::Point2f offset,
                float thickness, 
                const cv::Scalar& color,
                const float alpha){

        foxglove_msgs::msg::PointsAnnotation marker;
        marker.timestamp = frame_header.stamp;
        marker.thickness = thickness;
        marker.type = 1;

        foxglove_msgs::msg::Point2 pt;
        foxglove_msgs::msg::Color cr;

        cr.r = color[2]/255.0f;
        cr.g = color[1]/255.0f;
        cr.b = color[0]/255.0f;
        cr.a = alpha;

        for(int i = 0;i < pointLists.size();++i){
            #if BAYER_IMAGE
                pt.x = pointLists[i].x*2 + offset.x;
                pt.y = pointLists[i].y*2 + offset.y;
            #else
                pt.x = pointLists[i].x;
                pt.y = pointLists[i].y;
            #endif
            marker.points.emplace_back(pt);

            marker.outline_colors.emplace_back(cr);
        }
        return marker;

    }

    foxglove_msgs::msg::PointsAnnotation RotatedRectMarker(std_msgs::msg::Header& frame_header, const cv::RotatedRect& rotatedRect,cv::Point2f offset, float thickness, const cv::Scalar& color,const float alpha) {
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);

        foxglove_msgs::msg::PointsAnnotation marker;
        
        marker.timestamp = frame_header.stamp;
        marker.thickness = thickness;
        marker.type = 4;

        foxglove_msgs::msg::Point2 pt;
        foxglove_msgs::msg::Color cr;

        cr.r = color[2]/255.0f;
        cr.g = color[1]/255.0f;
        cr.b = color[0]/255.0f;
        cr.a = alpha;

        for (int i = 0; i < 4; ++i) {
            pt.x = vertices[i % 4].x + offset.x;
            pt.y = vertices[i % 4].y + offset.y;
            marker.points.emplace_back(pt);

            pt.x = vertices[(i+1) % 4].x + offset.x;
            pt.y = vertices[(i+1) % 4].y + offset.y;
            marker.points.emplace_back(pt);

            marker.outline_colors.emplace_back(cr);
        }
        return marker;
    }

    foxglove_msgs::msg::PointsAnnotation RectMarker(std_msgs::msg::Header& frame_header, 
                            const cv::Rect& rect,
                            cv::Point2f offset, 
                            float thickness, 
                            const cv::Scalar& color,
                            const float alpha){
        foxglove_msgs::msg::PointsAnnotation marker;
        cv::Point2f vertices[4] = {cv::Point2f(rect.x,rect.y),
                                   cv::Point2f(rect.x+rect.width,rect.y),
                                   cv::Point2f(rect.x+rect.width,rect.y+rect.height),
                                   cv::Point2f(rect.x,rect.y+rect.height)};
        marker.timestamp = frame_header.stamp;
        marker.thickness = thickness;
        marker.type = 4;

        foxglove_msgs::msg::Point2 pt;
        foxglove_msgs::msg::Color cr;

        cr.r = color[2]/255.0f;
        cr.g = color[1]/255.0f;
        cr.b = color[0]/255.0f;
        cr.a = alpha;

        for (int i = 0; i < 4; ++i) {
            pt.x = vertices[i % 4].x + offset.x;
            pt.y = vertices[i % 4].y + offset.y;
            marker.points.emplace_back(pt);

            pt.x = vertices[(i+1) % 4].x + offset.x;
            pt.y = vertices[(i+1) % 4].y + offset.y;
            marker.points.emplace_back(pt);

            marker.outline_colors.emplace_back(cr);
        }
        return marker;
    }

    foxglove_msgs::msg::TextAnnotation TextMarker(std_msgs::msg::Header& frame_header,
                    const cv::Point& pt,
                    std::string text,
                    float thickness, 
                    const cv::Scalar& color){
        foxglove_msgs::msg::TextAnnotation marker;

        marker.timestamp = frame_header.stamp;
        marker.font_size = thickness;

        marker.text = text;

        marker.text_color.r = color[2]/255.0f;
        marker.text_color.g = color[1]/255.0f;
        marker.text_color.b = color[0]/255.0f;
        marker.text_color.a = 1.0;

        marker.position.x = pt.x;
        marker.position.y = pt.y;

        return marker;

    }

    foxglove_msgs::msg::CircleAnnotation CircleMarker(std_msgs::msg::Header& frame_header,
                        const cv::Point& position,
                        float diameter,
                        float thickness, 
                        const cv::Scalar& color,
                        const float alpha){

        foxglove_msgs::msg::CircleAnnotation temp;

        temp.timestamp = frame_header.stamp;
        foxglove_msgs::msg::Point2 pt;
        pt.x = position.x;
        pt.y = position.y;
        temp.position = pt;
        temp.diameter = diameter;
        temp.thickness = thickness;

        foxglove_msgs::msg::Color cr;
        cr.r = color[2]/255.0f;
        cr.g = color[1]/255.0f;
        cr.b = color[0]/255.0f;
        cr.a = alpha;

        temp.outline_color = cr;

        return temp;

    }

    foxglove_msgs::msg::PointsAnnotation ArrowMarker(std_msgs::msg::Header& frame_header,
                const cv::Point& pt1,
                const cv::Point& pt2,
                float thickness, 
                const cv::Scalar& color){
        foxglove_msgs::msg::PointsAnnotation marker;
        marker.timestamp = frame_header.stamp;
        marker.thickness = thickness;
        marker.type = 4;

        foxglove_msgs::msg::Point2 pt,arrowPt1, arrowPt2;
        foxglove_msgs::msg::Color cr;

        cr.r = color[2]/255.0f;
        cr.g = color[1]/255.0f;
        cr.b = color[0]/255.0f;
        cr.a = 1.0;

        pt.x = pt1.x;
        pt.y = pt1.y;
        marker.points.emplace_back(pt);

        pt.x = pt2.x;
        pt.y = pt2.y;
        marker.points.emplace_back(pt);

        // 计算箭头角度和箭头两侧点的位置
        double angle = atan2(pt1.y - pt2.y, pt1.x - pt2.x);
        double arrowLength = 10.0 * thickness; // 箭头长度与线宽成正比
        
        // 计算箭头两侧点
        arrowPt1.x = pt2.x + arrowLength * cos(angle + CV_PI/6);
        arrowPt1.y = pt2.y + arrowLength * sin(angle + CV_PI/6);
        
        arrowPt2.x = pt2.x + arrowLength * cos(angle - CV_PI/6);
        arrowPt2.y = pt2.y + arrowLength * sin(angle - CV_PI/6);

        marker.points.emplace_back(pt);
        marker.points.emplace_back(arrowPt1);

        marker.points.emplace_back(pt);
        marker.points.emplace_back(arrowPt2);

        marker.outline_colors.emplace_back(cr);
        marker.outline_colors.emplace_back(cr);
        marker.outline_colors.emplace_back(cr);
        
        return marker;
    }

    void drawContour(cv::Mat& src,const std::vector<cv::Point>&contour,cv::Point2f offset, float thickness, const cv::Scalar& color){
        std::vector<cv::Point> temp = contour;

        if (offset != cv::Point2f(0, 0)) {
            std::for_each(temp.begin(), temp.end(), [&offset](cv::Point& pt) {
                pt.x += offset.x;
                pt.y += offset.y;
            });
        }
        cv::drawContours(src,{temp}, 0, color, thickness);
    }

    foxglove_msgs::msg::PointsAnnotation CrossMarker(std_msgs::msg::Header& frame_header,
                const cv::Point& pt,
                float thickness, 
                const cv::Scalar& color){
        
        foxglove_msgs::msg::PointsAnnotation temp;

        temp.type = 4;
        temp.timestamp = frame_header.stamp;
        temp.thickness = thickness;

        foxglove_msgs::msg::Point2 position1,position2,position3,position4;
        foxglove_msgs::msg::Color cr;

        position1.x = pt.x - 15.0f;
        position1.y = pt.y;
        position2.x = pt.x + 15.0f;
        position2.y = pt.y;
        position3.x = pt.x;
        position3.y = pt.y - 15.0f;
        position4.x = pt.x;
        position4.y = pt.y + 15.0f;

        cr.r = color[2]/255.0f;
        cr.g = color[1]/255.0f;
        cr.b = color[0]/255.0f;
        cr.a = 1.0;
        temp.outline_colors.emplace_back(cr);
        temp.outline_colors.emplace_back(cr);

        temp.points.emplace_back(position1);
        temp.points.emplace_back(position2);
        temp.points.emplace_back(position3);
        temp.points.emplace_back(position4);

        return temp;
    }

    void drawNumberedPoints(cv::Mat& image, const std::vector<cv::Point2f>& points, 
                       const cv::Scalar& color, 
                       int fontFace,
                       double fontScale, 
                       int thickness){
        // 遍历所有点并绘制编号
        for(size_t i = 0; i < points.size(); ++i) {
            // 获取当前点
            cv::Point2f pt = points[i];
            // 转换为整数坐标（绘图需要）
            cv::Point pt_int(static_cast<int>(pt.x), static_cast<int>(pt.y));
            // 创建数字文本
            std::string text = std::to_string(i + 1);  // 从1开始编号
            // 计算文本大小以居中
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
            // 水平居中：x = 圆心x - 文本宽度/2
            // 垂直居中：y = 圆心y + 文本高度/2 (因为OpenCV的y坐标向下增长)
            cv::Point textOrg(
                pt_int.x - textSize.width / 2,
                pt_int.y + textSize.height / 2
            );
            // 绘制背景圆（可选，提高数字可读性）
            cv::circle(image, pt_int, textSize.width/2 + 5, cv::Scalar(255, 255, 255), -1);
            cv::circle(image, pt_int, textSize.width/2 + 5, color, 2);
            // 绘制数字
            cv::putText(image, text, textOrg, fontFace, fontScale, color, thickness);
        }
    }

}//namespace buff_tracker