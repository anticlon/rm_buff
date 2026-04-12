#include "buff_tracker/SectorFinder.h"
#include "buff_tracker/Draw.h"
#include "buff_tracker/Utility.h"
//#include "buff_tracker/cudas.h"
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <console_bridge/console.h>
#include <cstdint>
#include <geometry_msgs/msg/vector3.hpp>
#include <memory>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <std_msgs/msg/detail/header__struct.hpp>
#include <string>
#include <unordered_map>

namespace buff_tracker {
    SectorFinder::SectorFinder(const rclcpp::NodeOptions &options)
        : Node("SectorFinder", options) {
        // OPENCV设置
        RCLCPP_INFO_STREAM(this->get_logger(), "OpenCV version: " << CV_VERSION);
        cv::setNumThreads(4);

        this->buffdata.angle = 0.0;

        RCLCPP_INFO(this->get_logger(), "SectorFinder Starting...");

        RCLCPP_INFO(this->get_logger(), "Init: Values...");
        values = std::make_unique<Values>(this);

        RCLCPP_INFO(this->get_logger(), "Init: EllipseFitter...");
        ef = std::make_unique<EllipseFitter>(values.get());

        RCLCPP_INFO(this->get_logger(), "Init: ROS interface...");
        sub_cam_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                "/image_bayer/camera_info", rclcpp::SensorDataQoS(),
                [this](sensor_msgs::msg::CameraInfo::SharedPtr cam) {
                    RCLCPP_INFO(this->get_logger(), "Received camera infos");
                    this->pnp = std::make_unique<PnPSolver>(values.get(), cam->k, cam->d);
                    sub_cam_info.reset();
                });

        RCLCPP_INFO(this->get_logger(), "Init: MemMapPool...");
        memMapPool = std::make_unique<MemMapPool>(this);

        // 初始化ROI和遮罩
        m_globalRoi = cv::Rect(0, 0, 1920, 1080);
        m_localMask = cv::Mat::zeros(1080, 1920, CV_8U);

        // 订阅比赛时间
        sub_time = this->create_subscription<ifr_interface::msg::SerialImuData>(
                "/serial/revice", rclcpp::SensorDataQoS(),
                [this](ifr_interface::msg::SerialImuData::SharedPtr data) {
                    this->remain_time = data->remain_time;
                    RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1e5, "Time update:" << this->remain_time);
                });

        // 订阅队伍信息
        sub_team = this->create_subscription<ifr_interface::msg::Team>(
                "/rm/self_team", rclcpp::SensorDataQoS(),
                [this](ifr_interface::msg::Team::SharedPtr team) {
                    this->team = team->team == 1 ? Team::BLUE : Team::RED;
                    RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1e5, "Team update:" << ((team->team == 1) ? "Blue" : "Red"));
                });

        // 订阅自瞄种类
        sub_aim_type = this->create_subscription<ifr_interface::msg::AimType>(
                "/rm/aim_type", rclcpp::SensorDataQoS(),
                [this](ifr_interface::msg::AimType::SharedPtr type) {
                    RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1e5, "Received type: " << int(type->type));
                    this->can_run = (type->type == 1) ? true : false;
                });

// 订阅相机节点
#if BAYER_IMAGE
        sub_src = this->create_subscription<ifr_interface::msg::BayerImage>("/image_bayer/image", rclcpp::SensorDataQoS(),
                                                                            std::bind(&SectorFinder::bayer_input, this, std::placeholders::_1));
        pub_back_ptr = this->create_publisher<decltype(msg_back_ptr)>("image_bayer/return_ptr", 10);
#else
        sub_src = this->create_subscription<sensor_msgs::msg::Image>("/camera/image_raw", rclcpp::SensorDataQoS(),
                                                                     std::bind(&SectorFinder::rgb_input, this, std::placeholders::_1));
#endif

        RCLCPP_INFO(this->get_logger(), "Init: Threshold Image...");
        pub_img_arrow = this->create_publisher<sensor_msgs::msg::Image>("/rm/buff/Threshold/arrow", rclcpp::SensorDataQoS());
        pub_img_armor = this->create_publisher<sensor_msgs::msg::Image>("/rm/buff/Threshold/armor", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "Init: Marker...");
        meshMarker.ns = "buff";
        meshMarker.action = visualization_msgs::msg::Marker::ADD;
        meshMarker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        meshMarker.scale.set__x(1.0).set__y(1.0).set__z(1.0);
        meshMarker.color.set__a(1.0).set__r(0.0).set__g(0.0).set__b(0.0);
        meshMarker.mesh_resource = "package://buff_tracker/meshes/buff_center.dae";

        arrowMarker.ns = "buff";
        arrowMarker.id = 1;
        arrowMarker.action = visualization_msgs::msg::Marker::ADD;
        arrowMarker.type = visualization_msgs::msg::Marker::ARROW;
        arrowMarker.scale.set__x(0.5).set__y(0.02).set__z(0.02);
        arrowMarker.color.set__a(1.0).set__r(0.0).set__g(1.0).set__b(0.0);

        lineMarker.ns = "buff";
        lineMarker.id = 2;
        lineMarker.action = visualization_msgs::msg::Marker::ADD;
        lineMarker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        lineMarker.color.set__a(1.0).set__r(0.0).set__g(1.0).set__b(0.0);
        lineMarker.scale.set__x(0.02).set__y(0.02).set__z(0.02);

        pub_markers = this->create_publisher<decltype(msg_markers)>("/rm/buff/markers", 10);
        pub_img_markers = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>("/rm/buff/ImgMarkers", rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "Init: OpenCV Preview...");
        pub_img_preview = this->create_publisher<sensor_msgs::msg::Image>("/rm/buff/preview", rclcpp::SensorDataQoS());

        pub_buffdata = this->create_publisher<rm_interface::msg::BuffData>("/rm/buff/buffData", 10);
    }

    void SectorFinder::rgb_input(const sensor_msgs::msg::Image::SharedPtr img_msg) {
        // 能量机关自瞄是否启动,获取bayerimage里面的各种数据
        if (!can_run) return;

        frame_header.stamp = img_msg->header.stamp;
        frame_header.frame_id = "camera_optical_frame";

        // 时间戳定义
        buffdata.header = frame_header;

        // OpenCV图片转ROS图片
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);
        cpu_src = cv_ptr->image;
        recognise(frame_header, cpu_src);//传入时间戳和图像数据
    }

    void SectorFinder::bayer_input(const ifr_interface::msg::BayerImage::SharedPtr msg) {
        // 能力机自瞄关是否启动
        if (!can_run) return;

        frame_header.stamp = msg->header.stamp;
        frame_header.frame_id = "camera_optical_frame";

        // 时间戳定义
        buffdata.header = frame_header;

        cpu_src = cv::Mat(msg->height, msg->width, CV_8UC1, reinterpret_cast<void *>(msg->data));
        recognise(frame_header, cpu_src, msg->data);
    }

    void SectorFinder::recognise(std_msgs::msg::Header &frame_header, cv::Mat &cpu_src, uint64_t data) {
        // 设置开始时间，记录数据收集时间
        rclcpp::Time total_start_time = this->now();

        //清理上一帧在画面上画的调试信息（框、文字等），准备画新的。
        img_markers.points.clear();
        img_markers.circles.clear();
        img_markers.texts.clear();

        //状态重置。默认认为还没找到目标。
        buffdata.has_target = false;
        buffdata.vaild_pnp = false;

        // 记录每个步骤的处理时间,用来存每一个子步骤（如预处理、PnP）分别花了多长时间。
        std::map<std::string, double> step_times;
        rclcpp::Time step_start;
        double step_time;

        try {
            // step0: 预处理
            step_start = this->now();
            step0_prepare();
            step_time = (this->now() - step_start).seconds() * 1000.0;
            step_times["预处理"] = step_time;
            step_statistics_["预处理"].total_time += step_time;
            step_statistics_["预处理"].frame_count++;
        } catch (const cv::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "步骤-0异常: %s", e.what());
            step_errorFix();
        }

#if BAYER_IMAGE
        // 返还bayer图像指针
        msg_back_ptr.ptr = data;
        pub_back_ptr->publish(msg_back_ptr);
#endif

        try {
            // step1-2: 轮廓查找
            step_start = this->now();
            step1_findContours();
            step2_goodLight();
            step_time = (this->now() - step_start).seconds() * 1000.0;
            step_times["轮廓查找"] = step_time;
            step_statistics_["轮廓查找"].total_time += step_time;
            step_statistics_["轮廓查找"].frame_count++;

        } catch (const cv::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "步骤-1异常: %s", e.what());
            step_errorFix();
            return;
        }

        step_start = this->now();
        if (step3_goodArrow() == false) {
            step_errorFix();
            step9_publish();
            return;
        }
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["灯条配对"] = step_time;
        step_statistics_["灯条配对"].total_time += step_time;
        step_statistics_["灯条配对"].frame_count++;

        // step4: ROI设置
        step_start = this->now();
        step4_setRoi();
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["ROI初始设置"] = step_time;
        step_statistics_["ROI初始设置"].total_time += step_time;
        step_statistics_["ROI初始设置"].frame_count++;

        // step5: 装甲板检测
        step_start = this->now();
        if (step5_detectArmor() == false) {
            step_errorFix();
            step9_publish();
            return;
        }
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["装甲板检测"] = step_time;
        step_statistics_["装甲板检测"].total_time += step_time;
        step_statistics_["装甲板检测"].frame_count++;


        // step6: R标中心检测
        step_start = this->now();
        if (step6_detectCenterR() == false) {
            step_errorFix();
            step9_publish();
            return;
        }
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["R标检测"] = step_time;
        step_statistics_["R标检测"].total_time += step_time;
        step_statistics_["R标检测"].frame_count++;

        // step: 帧PnP解算
        step_start = this->now();
        step_pnp();
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["PNP"] = step_time;
        step_statistics_["PNP"].total_time += step_time;
        step_statistics_["PNP"].frame_count++;

        // 椭圆拟合
        step_start = this->now();
        if (!step_ef()) {
            RCLCPP_INFO_STREAM(this->get_logger(), "EF:Ellipse Flittering(" << ef->getCurrentPointsCount() << '/' << values->MIN_NUM2FIT << ')');
            step9_publish();
            return;
        }
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["椭圆拟合"] = step_time;
        step_statistics_["椭圆拟合"].total_time += step_time;
        step_statistics_["椭圆拟合"].frame_count++;

        // 数据计算
        step_start = this->now();
        step_dataCal();
        step_time = (this->now() - step_start).seconds() * 1000.0;
        step_times["数据计算"] = step_time;
        step_statistics_["数据计算"].total_time += step_time;
        step_statistics_["数据计算"].frame_count++;

        // Roi设置
        step_start = this->now();
        step8_setRoi();
        step_times["ROI设置"] = step_time;
        step_statistics_["ROI设置"].total_time += step_time;
        step_statistics_["ROI设置"].frame_count++;

        // 结果发布
        step_start = this->now();
        step9_publish();
        step_times["消息发布"] = step_time;
        step_statistics_["消息发布"].total_time += step_time;
        step_statistics_["消息发布"].frame_count++;

#if DETAIL
        // 输出总时间和各步骤时间
        double total_time = (this->now() - total_start_time).seconds() * 1000.0;
        std::stringstream time_info;
        time_info << "处理总时间: " << std::fixed << std::setprecision(2) << total_time << "ms (";
        for (const auto &[step_name, time]: step_times) {
            time_info << step_name << ": " << time << "ms, ";
        }
        time_info << ")";

        // 计算并输出平均处理时间
        std::stringstream avg_time_info;
        avg_time_info << "平均处理时间: (";
        for (const auto &[step_name, stats]: step_statistics_) {
            if (stats.frame_count > 0) {
                double avg_time = stats.total_time / stats.frame_count;
                avg_time_info << step_name << ": " << std::fixed << std::setprecision(2) << avg_time << "ms, ";
            }
        }
        avg_time_info << ")";
        RCLCPP_INFO(this->get_logger(), "%s", time_info.str().c_str());
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "%s\n%s", time_info.str().c_str(), avg_time_info.str().c_str());

#endif
    }

    void SectorFinder::step0_prepare() {    
        // float src_size = static_cast<float>(cpu_src.size().area());
        // // --- 将内存映射到显存 ---
        //cv::cuda::GpuMatvoid *cuda_src_ptr = memMapPool->getDevicePointer(cpu_src.data, cpu_src.dataend - cpu_src.datastart);
        //gpu_src = cv::cuda::GpuMat(cpu_src.rows, cpu_src.cols, cpu_src.type(), cuda_src_ptr);

#if BAYER_IMAGE
        memMapPool->malloc(m_imageArrow, gpu_gray_arrow, cpu_src.size() / 2, CV_8UC1);
        memMapPool->malloc(m_imageArmor, gpu_gray_armor, cpu_src.size() / 2, CV_8UC1);
        if (team == Team::BLUE) {
            cudas::getColorFromBayerRG_hsv(
                    gpu_src, gpu_gray_arrow, gpu_gray_armor,
                    values->BLUE_CUDA_ARROW_HL, values->BLUE_CUDA_ARROW_HH,
                    values->BLUE_CUDA_ARROW_SL, values->BLUE_CUDA_ARROW_SH,
                    values->BLUE_CUDA_ARROW_VL, values->BLUE_CUDA_ARROW_VH,
                    values->BLUE_CUDA_ARMOR_HL, values->BLUE_CUDA_ARMOR_HH,
                    values->BLUE_CUDA_ARMOR_SL, values->BLUE_CUDA_ARMOR_SH,
                    values->BLUE_CUDA_ARMOR_VL, values->BLUE_CUDA_ARMOR_VH);
        } else {
            cudas::getColorFromBayerRG_hsv(
                    gpu_src, gpu_gray_arrow, gpu_gray_armor,
                    values->RED_CUDA_ARROW_HL, values->RED_CUDA_ARROW_HH,
                    values->RED_CUDA_ARROW_SL, values->RED_CUDA_ARROW_SH,
                    values->RED_CUDA_ARROW_VL, values->RED_CUDA_ARROW_VH,
                    values->RED_CUDA_ARMOR_HL, values->RED_CUDA_ARMOR_HH,
                    values->RED_CUDA_ARMOR_SL, values->RED_CUDA_ARMOR_SH,
                    values->RED_CUDA_ARMOR_VL, values->RED_CUDA_ARMOR_VH);
        }
#else
        //cudaDeviceSynchronize();
        // 分割通道
        // std::vector<cv::cuda::GpuMat> gpu_channels;
        // cv::cuda::split(gpu_src, gpu_channels);


        // 1. 将容器类型从 cv::cuda::GpuMat 改为 cv::Mat
        std::vector<cv::Mat> hsv_channels;
        // 2. 使用 CPU 版本的 cv::split，输入改为 CPU 端的 cv::Mat（即你之前代码中的 cpu_src）
        cv::cvtColor(cpu_src, cpu_hsv, cv::COLOR_BGR2HSV);
        cv::split(cpu_hsv, hsv_channels);
        cv::Mat H = hsv_channels[0];  // 色调通道 (Hue)
        cv::Mat S = hsv_channels[1];  // 饱和度通道 (Saturation)
        cv::Mat V = hsv_channels[2];  // 明度通道 (Value)
        // Assign channels to blue, green, and red for easier reference
        // blue = gpu_channels[2]; // Blue channel
        // green = gpu_channels[1];// Green channel
        // red = gpu_channels[0];  // Red channel

        // Create GPU memory for the thresholded images
        //memMapPool->malloc(m_imageArrow, gpu_gray_arrow, red.size(), CV_8UC1);// m_imageArrow:检测箭头用的二值化图片
        //memMapPool->malloc(m_imageArmor, gpu_gray_armor, red.size(), CV_8UC1);//m_imageArmor:检测装甲板边框用的二值化图片

        // Call the optimized color extraction function
        if (team == Team::BLUE) {
            // cudas::getColorFromRGB_hsv(//通过 HSV 空间提取 Arrow / Armor 掩码
            //         red,green,blue, gpu_gray_arrow, gpu_gray_armor,
            //         values->BLUE_CUDA_ARROW_HL, values->BLUE_CUDA_ARROW_HH,
            //         values->BLUE_CUDA_ARROW_SL, values->BLUE_CUDA_ARROW_SH,
            //         values->BLUE_CUDA_ARROW_VL, values->BLUE_CUDA_ARROW_VH,
            //         values->BLUE_CUDA_ARMOR_HL, values->BLUE_CUDA_ARMOR_HH,
            //         values->BLUE_CUDA_ARMOR_SL, values->BLUE_CUDA_ARMOR_SH,
            //         values->BLUE_CUDA_ARMOR_VL, values->BLUE_CUDA_ARMOR_VH);


            SectorFinder::extract_mask(
            //通过 HSV 空间提取 Arrow / Armor 掩码
                    H, S, V, cpu_gray_arrow, cpu_gray_armor,
                    values->BLUE_CUDA_ARROW_HL, values->BLUE_CUDA_ARROW_HH,
                    values->BLUE_CUDA_ARROW_SL, values->BLUE_CUDA_ARROW_SH,
                    values->BLUE_CUDA_ARROW_VL, values->BLUE_CUDA_ARROW_VH,
                    values->BLUE_CUDA_ARMOR_HL, values->BLUE_CUDA_ARMOR_HH,
                    values->BLUE_CUDA_ARMOR_SL, values->BLUE_CUDA_ARMOR_SH,
                    values->BLUE_CUDA_ARMOR_VL, values->BLUE_CUDA_ARMOR_VH     
            );
        } else {
            // cudas::getColorFromRGB_hsv(
            //         red,green,blue, gpu_gray_arrow, gpu_gray_armor,
            //         values->RED_CUDA_ARROW_HL, values->RED_CUDA_ARROW_HH,
            //         values->RED_CUDA_ARROW_SL, values->RED_CUDA_ARROW_SH,
            //         values->RED_CUDA_ARROW_VL, values->RED_CUDA_ARROW_VH,
            //         values->RED_CUDA_ARMOR_HL, values->RED_CUDA_ARMOR_HH,
            //         values->RED_CUDA_ARMOR_SL, values->RED_CUDA_ARMOR_SH,
            //         values->RED_CUDA_ARMOR_VL, values->RED_CUDA_ARMOR_VH);


            SectorFinder::extract_mask(
            //通过 HSV 空间提取 Arrow / Armor 掩码
                    H, S, V, cpu_gray_arrow, cpu_gray_armor,
                    values->RED_CUDA_ARROW_HL, values->RED_CUDA_ARROW_HH,
                    values->RED_CUDA_ARROW_SL, values->RED_CUDA_ARROW_SH,
                    values->RED_CUDA_ARROW_VL, values->RED_CUDA_ARROW_VH,
                    values->RED_CUDA_ARMOR_HL, values->RED_CUDA_ARMOR_HH,
                    values->RED_CUDA_ARMOR_SL, values->RED_CUDA_ARMOR_SH,
                    values->RED_CUDA_ARMOR_VL, values->RED_CUDA_ARMOR_VH   
            );
        }
#endif
        // 取ROI
        m_imageArrow = m_imageArrow(m_globalRoi);
        m_imageArmor = m_imageArmor(m_globalRoi);

        // 生成全局ROI预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.points.emplace_back(RectMarker(frame_header, RoiX2(m_globalRoi), cv::Point2f(0, 0), 3, cv::Scalar(0, 255, 0)));
#else
            img_markers.points.emplace_back(RectMarker(frame_header, m_globalRoi, cv::Point2f(0, 0), 3, cv::Scalar(0, 255, 0)));
#endif
        }

        // 生成流水灯阈值图预览
        if (pub_img_arrow->get_subscription_count()) {
            auto msg_arrow = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", m_imageArrow.clone()).toImageMsg();
            pub_img_arrow->publish(*msg_arrow);
        }
        // 生成击打板阈值图预览
        if (pub_img_armor->get_subscription_count()) {
            auto msg_armor = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", m_imageArmor.clone()).toImageMsg();
            pub_img_armor->publish(*msg_armor);
        }
        // 生成原图像预览
        if (pub_img_preview->get_subscription_count() && buffdata.has_target) {
#if BAYER_IMAGE
            auto msg_src = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", cpu_src).toImageMsg();
#else
            auto msg_src = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", cpu_src).toImageMsg();
#endif
            pub_img_preview->publish(*msg_src);
        }
    }

    void SectorFinder::step1_findContours() {
        // 轮廓查找
        cv::findContours(m_imageArrow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }

    void SectorFinder::step2_goodLight() {
        // 查找流水灯灯条
        lightlines.clear();
        for (const auto &i: contours) {
#if BAYER_IMAGE
            Lightline lightline(i, m_globalRoi, {0, 0, (float) (cpu_src.cols / 2.0), (float) (cpu_src.rows / 2.0)});
#else
            Lightline lightline(i, m_globalRoi, {0, 0, (float) cpu_src.cols, (float) cpu_src.rows});
#endif
            // 面积过滤
            if (!inRange(lightline.m_area, values->MIN_ARROW_LIGHTLINE_AREA, values->MAX_ARROW_LIGHTLINE_AREA)) continue;
            // 长宽比过滤
            if (lightline.m_aspectRatio > values->MAX_ARROW_LIGHTLINE_ASPECT_RATIO) continue;
            // 都符合存入lightlines
            lightlines.emplace_back(std::move(lightline));
        }

        if (pub_img_markers->get_subscription_count()) {
            for (const auto &i: lightlines) {
#if BAYER_IMAGE
                img_markers.points.emplace_back(RotatedRectMarker(frame_header, RRX2(i.m_rotatedRect), m_globalRoi.tl() * 2, 3, cv::Scalar(0, 255, 0), 0.3));// 流水灯条预览
#else
                img_markers.points.emplace_back(RotatedRectMarker(frame_header, i.m_rotatedRect, m_globalRoi.tl(), 3, cv::Scalar(0, 255, 0), 0.6));  // 流水灯条预览
#endif
            }
        }
    }

    bool SectorFinder::step3_goodArrow() {
        // 利用 cv::partition 匹配箭头
        std::vector<int> labels;
        cv::partition(lightlines, labels, sameArrow);//sameArrow函数的用法：比较两个灯条是否满足在一个箭头内的条件，是则返回 true，否则为 false
                                                     //根据自定义函数分类lightlines
        // 统计每个 label 出现次数
        std::unordered_map<int, int> data;
        for (auto label: labels) {
            data[label]++;
        }
        if (data.empty()) {
            RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Arrow:Empty List");
            return false;
        }
        // 寻找出现次数最多的 label
        auto max_iter = std::max_element(data.begin(), data.end(),
                                         [](const std::pair<int, int> &i, const std::pair<int, int> &j) {
                                             return i.second < j.second;
                                         });
        int maxLabel = max_iter->first;
        int maxNum = max_iter->second;
        // 判断 num 是否符合要求
        if (!inRange(maxNum, values->MIN_ARROW_LIGHTLINE_NUM, values->MAX_ARROW_LIGHTLINE_NUM)) {
            RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Arrow:NUM_NOT_MATCH");
#if DETAIL
            RCLCPP_WARN_STREAM(this->get_logger(), "Max" << values->MAX_ARROW_LIGHTLINE_NUM << " Min:" << values->MIN_ARROW_LIGHTLINE_NUM << " Val:" << maxNum);
#endif
            return false;
        }
        // 获取和 maxLabel 相同的灯条
        std::vector<int> arrowIndices;
        for (unsigned int i = 0; i < labels.size(); ++i) {
            if (labels[i] == maxLabel) {
                arrowIndices.emplace_back(i);
            }
        }
        // 通过索引将对应的灯条加入箭头
        std::vector<Lightline> arrowLightlines;
        for (auto index: arrowIndices) {
            arrowLightlines.emplace_back(lightlines.at(index));
        }
        //Arrow::set的作用
        // 把两个灯条粘成一个箭头
        // 拟合中轴线，过滤噪点
        // 套旋转矩形，拿到形状
        // 修正角度，保证长宽正确
        // 计算判断条件，用于筛选真假箭头
        m_arrow.set(arrowLightlines, m_globalRoi.tl());
        // 判断长宽比
        if (!inRange(m_arrow.m_aspectRatio, values->MIN_ARROW_ASPECT_RATIO, values->MAX_ARROW_ASPECT_RATIO)) {
            RCLCPP_WARN(this->get_logger(), "Arrow:ASPECT_RATIO_NOT_MATCH ");
#if DETAIL
            RCLCPP_WARN_STREAM(this->get_logger(), "Max:" << values->MAX_ARROW_ASPECT_RATIO << " Min:" << values->MIN_ARROW_ASPECT_RATIO << " Val:" << m_arrow.m_aspectRatio);
#endif
            return false;
        }
        // 判断面积
        if (m_arrow.m_area > values->MAX_ARROW_AREA) {
            RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Arrow:AREA_NOT_MATCH ");
#if DETAIL
            RCLCPP_WARN_STREAM(this->get_logger(), "Max:" << values->MAX_ARROW_AREA << " Val:" << m_arrow.m_area);
#endif
            return false;
        }
        // 生成流水灯预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.points.emplace_back(RotatedRectMarker(frame_header, RRX2(m_arrow.m_rotatedRect), m_globalRoi.tl() * 2, 3, cv::Scalar(0, 255, 0), 0.6));// 流水灯条预览
#else
            img_markers.points.emplace_back(RotatedRectMarker(frame_header, m_arrow.m_rotatedRect, m_globalRoi.tl(), 3, cv::Scalar(0, 255, 0), 0.6));// 流水灯条预览
#endif
        }
        return true;
    }

    void SectorFinder::step4_setRoi() {
        // 设置ROI矩形
        double distance = m_arrow.m_length * values->LOCAL_ROI_DISTANCE_RATIO;
        float width = values->LOCAL_ROI_LENTH;
        // 计算ROI中心点
        float x = distance * std::cos(angle2Radian(m_arrow.m_angle));
        float y = distance * std::sin(angle2Radian(m_arrow.m_angle));
        cv::Point2f centerUp{m_arrow.m_center.x - m_globalRoi.x + x, m_arrow.m_center.y - m_globalRoi.y + y};
        cv::Point2f centerDown{m_arrow.m_center.x - x - m_globalRoi.x, m_arrow.m_center.y - m_globalRoi.y - y};
        // 创建旋转矩形
        cv::RotatedRect rectUp{centerUp, cv::Size(width, width), (float) m_arrow.m_angle};
        cv::RotatedRect rectDown{centerDown, cv::Size(width, width), (float) m_arrow.m_angle};

        std::array<std::array<cv::Point2f, 4>, 2> roiPoints;
        rectUp.points(roiPoints.at(0).begin());
        rectDown.points(roiPoints.at(1).begin());
        // 防止越界
        for (auto &points: roiPoints) {
            for (auto &point: points) {
                if (point.x < 0) point.x = 0;
                if (point.x > m_globalRoi.width) point.x = m_globalRoi.width;
                if (point.y < 0) point.y = 0;
                if (point.y > m_globalRoi.height) point.y = m_globalRoi.height;
            }
        }
        // 填充ROI掩码
        for (const auto &points: roiPoints) {
            std::vector<cv::Point> _points;
            for (const auto &point: points) {
                _points.emplace_back(static_cast<cv::Point>(point));
            }
            cv::fillConvexPoly(m_localMask, _points, cv::Scalar(255, 255, 255));
        }
        // 调整ROI不超出图像边界
        m_armorRoi = cv::Rect2f(centerUp.x - width * 0.5, centerUp.y - width * 0.5, width, width);
        m_centerRoi = cv::Rect2f(centerDown.x - width * 0.5, centerDown.y - width * 0.5, width, width);
        resetRoi(m_armorRoi, m_globalRoi);
        resetRoi(m_centerRoi, m_globalRoi);
        // 判断是否需要交换ROI
        cv::Rect2f centerRoiGlobal{m_centerRoi.x + m_globalRoi.x, m_centerRoi.y + m_globalRoi.y, m_centerRoi.width, m_centerRoi.height};
        if (!inRect(m_centerR.m_center, centerRoiGlobal)) {
            std::swap(m_armorRoi, m_centerRoi);
        }
        // 生成两张ROI预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.points.emplace_back(RectMarker(frame_header, RoiX2(m_armorRoi), m_globalRoi.tl() * 2, 3, cv::Scalar(255, 0, 0), 0.6)); // 装甲板ROI预览
            img_markers.points.emplace_back(RectMarker(frame_header, RoiX2(m_centerRoi), m_globalRoi.tl() * 2, 3, cv::Scalar(0, 255, 0), 0.6));// 中心ROI预览
#else
            img_markers.points.emplace_back(RectMarker(frame_header, m_armorRoi, m_globalRoi.tl(), 3, cv::Scalar(255, 0, 0), 0.6));                  // 装甲板ROI预览
            img_markers.points.emplace_back(RectMarker(frame_header, m_centerRoi, m_globalRoi.tl(), 3, cv::Scalar(0, 255, 0), 0.6));                 // 中心ROI预览
#endif
        }
    }

    bool SectorFinder::step5_detectArmor() {
        // armor roi 区域的图像为检测图像，center roi 区域为备用图像
        cv::Mat detect = (m_imageArmor & m_localMask)(m_armorRoi);
        cv::Mat backup = (m_imageArmor & m_localMask)(m_centerRoi);

        // 调换标志位，如果检测不到，则调换检测图像和备用图像，并将其置为 true
        bool reverse = false;
        TargetShape TargetCross;
        std::vector<std::string> err;
    RESTART:
        // 寻找符合装甲板边框要求的灯条
        if (findArmorTargetShape(detect, TargetCross, err, m_globalRoi, m_armorRoi) == false) {
            // 如果找不到并且已经调换过图像了，则检测失败
            if (reverse == true) {
                RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Armor:Not Found");
#if DETAIL
                RCLCPP_WARN_STREAM(this->get_logger(), "P1_Err:[" << err[0] << "]");
                RCLCPP_WARN_STREAM(this->get_logger(), "P2_Err:[" << err[1] << "]");
#endif
                return false;
            }
            // 如果找不到并且没有调换过，则调换图像并置标志位
            std::swap(detect, backup);
            std::swap(m_armorRoi, m_centerRoi);
            reverse = true;
            // 回到检测装甲板灯条处
            goto RESTART;
        }
        m_armor.set(TargetCross);
        // 生成装甲板中心预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.points.emplace_back(CrossMarker(frame_header, m_armor.m_targetshape.m_center * 2, 3, cv::Scalar(0, 255, 0)));// 装甲板中心预览
#else
            img_markers.points.emplace_back(CrossMarker(frame_header, m_armor.m_targetshape.m_center, 3, cv::Scalar(0, 255, 0)));                    // 装甲板中心预览
#endif
        }
        return true;
    }

    bool SectorFinder::step6_detectCenterR() {
        m_imageCenter = (m_imageArmor & m_localMask)(m_centerRoi);
        // 寻找中心灯条，可能是多个
        std::vector<Lightline> lightlines;
        if (findCenterLightlines(m_imageCenter, lightlines, m_globalRoi, m_centerRoi) == false) {
            RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Center:Empty List");
            return false;
        }
        // 从灯条中寻找中心 R
        if (findCenterR(m_centerR, lightlines, m_arrow, m_armor) == false) {
            RCLCPP_WARN_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Center:Not Found");
            return false;
        }
        // 生成中心R预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.points.emplace_back(CrossMarker(frame_header, m_centerR.m_center * 2, 3, cv::Scalar(0, 255, 0)));// R中心预览
#else
            img_markers.points.emplace_back(CrossMarker(frame_header, m_centerR.m_center, 3, cv::Scalar(0, 255, 0)));                                // R中心预览
#endif
        }
        return true;
    }

    bool SectorFinder::step_pnp() {
        if (pnp != nullptr) {
            // 提取装甲板标定点
            cameraPoints = getCameraPoints(m_armor, m_centerR);
            if (cameraPoints.empty()) {
                RCLCPP_ERROR(this->get_logger(), "pnp:chileren num err : %d", m_armor.m_targetshape.m_children);
                return false;
            }
            if (!pnp->solvePnP(cameraPoints, buffdata.pose)) {
                RCLCPP_ERROR(this->get_logger(), "Pnp:bad point");
                return false;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "pnp:not init...");
            return false;
        }
        //生成PNP预览
        if (pub_markers->get_subscription_count()) {
            meshMarker.pose = buffdata.pose;
            meshMarker.header = frame_header;
            meshMarker.id = 5;
            meshMarker.mesh_resource = "package://buff_tracker/meshes/buff_center.dae";
            msg_markers.markers.emplace_back(meshMarker);
        }
        buffdata.vaild_pnp = true;
        return true;
    }

    void SectorFinder::step8_setRoi() {
        // 计算下张图片中的能量机关ROI
        double width{values->GLOBAL_ROI_LENGTH_RATIO * 2 *
                     pointPointDistance(m_armor.m_center, m_centerR.m_center)};
        m_globalRoi = cv::Rect2f(m_centerR.m_x - 0.5 * width, m_centerR.m_y - 0.5 * width, width, width);
// 防止越界
#if BAYER_IMAGE
        resetRoi(m_globalRoi, cpu_src.rows / 2, cpu_src.cols / 2);
#else
        resetRoi(m_globalRoi, cpu_src.rows, cpu_src.cols);
#endif
    }

    bool SectorFinder::step_ef() {
        // 将装甲板中心加入拟合点集
        this->ef->addPointWithDistanceCheck(m_armor.m_center - m_centerR.m_center);
        // 若点集内的点过少，则椭圆置信度低
        if (ef->getCurrentPointsCount() < values->MIN_NUM2FIT) {
            if (pub_img_markers->get_subscription_count()) {
                img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 25)) * 2, "椭圆拟合状态", 40, cv::Scalar(0, 255, 0)));
                img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 50)) * 2,
                                                          "最低拟合点数:" + std::to_string(ef->getCurrentPointsCount()) + "/" + std::to_string(values->MIN_NUM2FIT), 40, cv::Scalar(0, 255, 0)));
            }
            return false;
        }
        // 尝试进行拟合
        this->ef->fitStandard();
        // 若椭圆未完成拟合，则退出
        if (!ef->isFit()) return false;
        // 生成椭圆拟合信息预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 25)) * 2,
                                                      "椭圆拟合状态",
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 50)) * 2,
                                                      "点数:" + std::to_string(ef->getCurrentPointsCount()) + "/" + std::to_string(values->MAX_POINTS),
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 75)) * 2,
                                                      "置信度:" + std::to_string(ef->getFitQuality().confidence * 100.0f) + "%",
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 100)) * 2,
                                                      "内点比例:" + std::to_string(ef->getFitQuality().inlier_ratio * 100.0f) + "%",
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 125)) * 2,
                                                      "实际迭代次数:" + std::to_string(ef->getFitQuality().iterations),
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.points.emplace_back(PointsMarker(frame_header, ef->getInliers(), m_centerR.m_center * 2, 3, cv::Scalar(0, 255, 0), 0.7)); // 内点绘制
            img_markers.points.emplace_back(PointsMarker(frame_header, ef->getOutliers(), m_centerR.m_center * 2, 3, cv::Scalar(0, 0, 255), 0.7));// 离群点绘制
#else
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 25)),
                                                      "椭圆拟合状态",
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 50)),
                                                      "点数:" + std::to_string(ef->getCurrentPointsCount()) + "/" + std::to_string(values->MAX_POINTS),
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 75)),
                                                      "置信度:" + std::to_string(ef->getFitQuality().confidence * 100.0f) + "%",
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 100)),
                                                      "内点比例:" + std::to_string(ef->getFitQuality().inlier_ratio * 100.0f) + "%",
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() + cv::Point2f(2, 125)),
                                                      "实际迭代次数:" + std::to_string(ef->getFitQuality().iterations),
                                                      40,
                                                      cv::Scalar(0, 255, 0)));
            img_markers.points.emplace_back(PointsMarker(frame_header, ef->getInliers(), m_centerR.m_center, 3, cv::Scalar(0, 255, 0), 0.7)); // 内点绘制
            img_markers.points.emplace_back(PointsMarker(frame_header, ef->getOutliers(), m_centerR.m_center, 3, cv::Scalar(0, 0, 255), 0.7));// 离群点绘制
#endif
        }
        return true;
    }

    void SectorFinder::step_dataCal() {
        // 单位向量计算
        auto reVec = calculateUnitVector(ef->mapPoint(m_armor.m_center - m_centerR.m_center));
        //auto reVec = calculateDirectionVector(m_arrow.m_rotatedRect,m_centerR.m_center);
        buffdata.dvx = reVec.x;
        buffdata.dvy = reVec.y;

        // 映射相关预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.circles.emplace_back(CircleMarker(frame_header, m_centerR.m_center * 2 + ef->getCenter() * 2, ef->getCircumscribedRadius() * 4.0f, 3, cv::Scalar(0, 255, 255), 0.5));// 映射圆
            img_markers.points.emplace_back(ArrowMarker(frame_header, m_centerR.m_center * 2,
                                                        ef->mapPoint(m_armor.m_center - m_centerR.m_center) * 2 + m_centerR.m_center * 2, 5, cv::Scalar(0, 255, 255)));// 映射方向
#else
            img_markers.circles.emplace_back(CircleMarker(frame_header, m_centerR.m_center, ef->getCircumscribedRadius() * 2.0f, 3, cv::Scalar(0, 255, 255), 0.5));
            img_markers.points.emplace_back(CrossMarker(frame_header, m_centerR.m_center + ef->mapPoint(m_armor.m_center - m_centerR.m_center), 3, cv::Scalar(0, 255, 255)));
#endif
        }

        // 计算打击点
        geometry_msgs::msg::Pose temp, target;
        temp = rotatePoseWithEulerAngles(buffdata.pose, 0, -0.5 * CV_PI, 0.5 * CV_PI);
        target.position = moveAlongPoseDirection(temp, 0.7f);
        target.orientation = buffdata.pose.orientation;

        if (pub_markers->get_subscription_count()) {
            arrowMarker.pose = target;
            msg_markers.markers.emplace_back(arrowMarker);
        }
        // 延时计算
        if (buffdata.vaild_pnp) {
            const double g = 9.80665;///< 重力加速度
            double v0 = 25.0;
            double g_v02 = g / (v0 * v0);
            const auto X2 = std::pow(buffdata.pose.position.x, 2) +
                            std::pow(buffdata.pose.position.y, 2);
            const auto Y = -buffdata.pose.position.z;
            const auto l = std::sqrt(X2 + std::pow(Y, 2));
            const auto sin0 = Y / l;
            const auto a = (std::asin(sin0) - std::asin((g_v02 * l * (1 - std::pow(sin0, 2))) - sin0)) / 2.0;
            const auto X = std::sqrt(X2);
            const auto t = X / (std::cos(a) * v0);
            buffdata.delay = t + values->DELTA_TIME;
        }

        // 跳变计算
        buffdata.update = false;
        double CiOUPoint = -99;
        std::vector<cv::Point> temp_contour{m_armor.m_targetshape.m_tl, m_armor.m_targetshape.m_tr,
                                            m_armor.m_targetshape.m_bl, m_armor.m_targetshape.m_br};
        cv::Rect rect = cv::boundingRect(temp_contour);
        if (prev_available) CiOUPoint = calculateCIoU(prev_armor, rect);
        if (CiOUPoint < 0.1) buffdata.update = true;
        if (miss_target > 4) buffdata.update = true;

        // 大小符模式判定
        switch (values->BUFF_MODE) {
            case 0:
                buffdata.mode = 0;
                if (this->remain_time < uint16_t(240)) buffdata.mode = 1;
                break;
            case 1:
                buffdata.mode = 0;
                break;
            case 2:
                buffdata.mode = 1;
                break;
        }

        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() - cv::Point2f(0, 25)) * 2,
                                                      buffdata.mode == 0 ? "小能量机关" : "大能量机关",
                                                      55, cv::Scalar(0, 255, 0)));// 能量机关状态预览
#else
            img_markers.texts.emplace_back(TextMarker(frame_header, m_globalRoi.tl() - cv::Point2f(0, 25), buffdata.mode == 0 ? "小能量机关" : "大能量机关", 55, cv::Scalar(0, 255, 0)));// 能量机关状态预览
#endif
        }

        // 输入其他信息
        buffdata.rx = m_centerR.m_center.x;
        buffdata.ry = m_centerR.m_center.y;
        buffdata.rad = ef->getCircumscribedRadius();

        // 是否有目标
        buffdata.has_target = true;

        // 存储上一帧关键数据
        prev_armor = rect;
        prev_centerR = m_centerR.m_boundingRect;
        prev_available = true;
    }

    void SectorFinder::step9_publish() {
        // 发布数据
        pub_buffdata->publish(buffdata);
        if (!buffdata.has_target) {
            miss_target++;
        } else {
            miss_target = 0;
        }
        /*
        if (buffdata.has_target){
            RCLCPP_INFO(this->get_logger(),"x:%14.8f y:%14.8f time[%ld.%09ld] update:%d mode:%d delay:%.6lf s",
                buffdata.dvx,buffdata.dvy,
                buffdata.header.stamp.sec,buffdata.header.stamp.nanosec,
                buffdata.update,buffdata.mode,buffdata.delay);
        }
        */
        // 能量机关跟踪状态预览
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.texts.emplace_back(TextMarker(frame_header, (m_globalRoi.tl() - cv::Point2f(60, 25)) * 2,
                                                      buffdata.has_target ? "追踪" : "丢失",
                                                      55, cv::Scalar(0, 255, 0)));
#else
            img_markers.texts.emplace_back(TextMarker(frame_header, m_globalRoi.tl() - cv::Point2f(60, 25),
                                                      buffdata.has_target ? "追踪" : "丢失",
                                                      55, cv::Scalar(0, 255, 0)));
#endif
            pub_img_markers->publish(img_markers);
        }
        if (pub_markers->get_subscription_count() && buffdata.vaild_pnp) {
            // 辅助线
            lineMarker.points.clear();
            geometry_msgs::msg::Point yd;
            yd.set__x(0).set__y(0).set__z(0);
            lineMarker.points.emplace_back(yd);
            lineMarker.points.emplace_back(buffdata.pose.position);
            lineMarker.header = frame_header;
            //msg_markers.markers.emplace_back(lineMarker);
            // 打击点法向量
            arrowMarker.header = frame_header;
            msg_markers.markers.emplace_back(arrowMarker);
            pub_markers->publish(msg_markers);
            msg_markers.markers.clear();
        }
    }

    void SectorFinder::step_errorFix() {
// 如果检测失败，则将全局 roi 设为和原图片一样大小
#if BAYER_IMAGE
        m_globalRoi = cv::Rect(0, 0, cpu_src.cols / 2, cpu_src.rows / 2);
        m_localMask = cv::Mat::zeros(cpu_src.rows / 2, cpu_src.cols / 2, CV_8U);
#else
        m_globalRoi = cv::Rect(0, 0, cpu_src.cols, cpu_src.rows);
        m_localMask = cv::Mat::zeros(cpu_src.rows, cpu_src.cols, CV_8U);
#endif
        //RCLCPP_INFO_STREAM(this->get_logger(),"size "<< cpu_src.cols << " " << cpu_src.rows);
    }

    void SectorFinder::extract_mask(const cv::Mat& h,   const cv::Mat& s,   const cv::Mat& v,
            cv::Mat& dst_arrow, cv::Mat& dst_armor,
            // Arrow HSV 范围
            int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
            // Armor HSV 范围
            int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m){
    
    
                // 1. 输入校验：确保H/S/V尺寸一致、均为单通道8位图像
    CV_Assert(h.size() == s.size() && h.size() == v.size());
    CV_Assert(h.type() == CV_8UC1 && s.type() == CV_8UC1 && v.type() == CV_8UC1);

    // 2. 初始化输出掩码：全黑（0），尺寸和输入一致
    dst_arrow = cv::Mat::zeros(h.size(), CV_8UC1);
    dst_armor = cv::Mat::zeros(h.size(), CV_8UC1);

    // 3. 遍历所有像素（i=行，j=列）
    for (int i = 0; i < h.rows; i++)
    {
        for (int j = 0; j < h.cols; j++)
        {
            // 获取当前像素的H、S、V值（单通道，用uchar访问）
            uchar h_val = h.at<uchar>(i, j);
            uchar s_val = s.at<uchar>(i, j);
            uchar v_val = v.at<uchar>(i, j);

            // ---------------- 箭头掩码判断 ----------------
            bool is_arrow = (h_val >= h_low_a && h_val <= h_high_a) &&
                            (s_val >= s_low_a && s_val <= s_high_a) &&
                            (v_val >= v_low_a && v_val <= v_high_a);
            if (is_arrow)
            {
                dst_arrow.at<uchar>(i, j) = 255; // 满足条件：白色
            }

            // ---------------- 装甲板掩码判断 ----------------
            bool is_armor = (h_val >= h_low_m && h_val <= h_high_m) &&
                            (s_val >= s_low_m && s_val <= s_high_m) &&
                            (v_val >= v_low_m && v_val <= v_high_m);
            if (is_armor)
            {
                dst_armor.at<uchar>(i, j) = 255; // 满足条件：白色
            }
        }
    }
                
        
    }

    /**
     * @brief 寻找符合装甲板要求的边框灯条，并将其存入一个向量中。成功返回 true，否则返回false。
     * @param[in] image         二值图
     * @param[in] TargetCross    边框灯条
     * @param[in] err           调试信息
     * @param[in] globalRoi     全局roi，用来设置灯条的正确位置
     * @param[in] localRoi      局部roi，用来设置灯条的正确位置
     * @return true
     * @return false
     */
    bool SectorFinder::findArmorTargetShape(const cv::Mat &image, TargetShape &TargetCross, std::vector<std::string> &err,
                                            const cv::Rect2f &globalRoi, const cv::Rect2f &localRoi) {
        // 寻找轮廓
        std::vector<TargetShape> GoodShapes;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); ++i) {
            // 必须有子层级
            if (hierarchy[i][2] == -1) {
                continue;
            }
            // 全部符合条件，则存入该灯条
            GoodShapes.emplace_back(std::move(TargetShape(contours, hierarchy, i, globalRoi, localRoi)));
        }
        // 如果符合要求的装甲板数目为空，则检测失败
        if (GoodShapes.empty()) {
            err.emplace_back("Empty List");
            return false;
        }
        // 根据贴合度排名
        auto BetterShape = std::max_element(GoodShapes.begin(), GoodShapes.end(), [](TargetShape t1, TargetShape t2) { return t1.m_counterArea < t2.m_counterArea; });
        // 判断轮廓面积
        if (inRange(BetterShape->m_counterArea, values->MIN_ARMOR_TARGETSHAPE_CONTOUR_AREA,
                    values->MAX_ARMOR_TARGETSHAPE_CONTOUR_AREA) == false) {
            err.emplace_back(("Area Not Match Val:" + std::to_string(static_cast<int>(BetterShape->m_counterArea)) +
                              " Max:" + std::to_string(static_cast<int>(values->MAX_ARMOR_TARGETSHAPE_CONTOUR_AREA)) +
                              " Min:" + std::to_string(static_cast<int>(values->MIN_ARMOR_TARGETSHAPE_CONTOUR_AREA))));
            return false;
        }
        // 判断长宽比
        if (BetterShape->m_aspectRatio > values->MAX_ARMOR_TARGETSHAPE_ASPECT_RATIO) {
            err.emplace_back(("Ratio Not Match Val:" + std::to_string(static_cast<float>(BetterShape->m_aspectRatio)) +
                              " Max:" + std::to_string(static_cast<float>(values->MAX_ARMOR_TARGETSHAPE_ASPECT_RATIO))));
            return false;
        }
        // 判断子轮廓数量
        if (inRange(BetterShape->m_children, values->MIN_ARMOR_TARGETSHAPE_CHILDREN_NUM,
                    values->MAX_ARMOR_TARGETSHAPE_CHILDREN_NUM) == false) {
            err.emplace_back(("Children Num Not Match Val:" + std::to_string(static_cast<int>(BetterShape->m_children)) +
                              " Max:" + std::to_string(static_cast<int>(values->MAX_ARMOR_TARGETSHAPE_CHILDREN_NUM)) +
                              " Min:" + std::to_string(static_cast<int>(values->MIN_ARMOR_TARGETSHAPE_CHILDREN_NUM))));
            return false;
        }
        TargetCross = *BetterShape;
        return true;
    }

    /**
     * @brief 寻找符合中心 R 要求的中心灯条，并将其存入一个向量中。成功返回 true，否则返回
     * false。
     * @param[in] image         带有中心区域的图片
     * @param[in] lightlines    存储的灯条向量
     * @param[in] globalRoi     全局 roi
     * @param[in] localRoi      局部 roi
     * @return true
     * @return false
     */
    bool SectorFinder::findCenterLightlines(const cv::Mat &image, std::vector<Lightline> &lightlines,
                                            const cv::Rect2f &globalRoi, const cv::Rect2f &localRoi) {
        // 寻找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const auto &contour: contours) {
            Lightline lightline(contour, globalRoi, localRoi);

            // 判断面积
            if (inRange(lightline.m_area, values->MIN_CENTER_AREA, values->MAX_CENTER_AREA) == false) {
                continue;
            }
            // 判断长宽比
            if (lightline.m_aspectRatio > values->MAX_CENTER_ASPECT_RATIO) {
                continue;
            }
            // 如果全部符合，则存入向量中
            lightlines.emplace_back(std::move(lightline));
        }
        // 符合要求灯条的数量为 0 则失败
        if (lightlines.empty()) {
            return false;
        }
        return true;
    }

    /**
     * @brief 根据中心灯条寻找并设置中心，成功返回 true，失败返回 false
     * @param[in] center        中心
     * @param[in] lightlines    中心灯条向量
     * @param[in] arrow         箭头
     * @param[in] armor         装甲板
     * @return true
     * @return false
     */
    bool SectorFinder::findCenterR(CenterR &center, const std::vector<Lightline> &lightlines, const Arrow &arrow,
                                   const Armor &armor) {
        // 设置中心 R 到装甲板中心的距离范围
        const double distance2ArmorCenter{(armor.m_targetshape.m_length + armor.m_targetshape.m_width) *
                                          values->POWER_RUNE_RADIUS /
                                          (values->ARMOR_TARGETSHAPE_RADIUS * 2.0)};
        const double ratio = 0.85;
        const double maxDistance2ArmorCenter{distance2ArmorCenter / ratio};
        const double minDistance2ArmorCenter{distance2ArmorCenter * ratio};
        // 设置中心 R 到箭头所在直线的最大距离
        const double maxDistance2ArrowLine{0.3 * armor.m_targetshape.m_width};
        if (pub_img_markers->get_subscription_count()) {
#if BAYER_IMAGE
            img_markers.circles.emplace_back(CircleMarker(frame_header, armor.m_targetshape.m_center * 2, maxDistance2ArmorCenter * 4.0f, 3, cv::Scalar(0, 255, 0), 0.3));
            img_markers.circles.emplace_back(CircleMarker(frame_header, armor.m_targetshape.m_center * 2, minDistance2ArmorCenter * 4.0f, 3, cv::Scalar(0, 255, 0), 0.3));
            img_markers.circles.emplace_back(CircleMarker(frame_header, armor.m_targetshape.m_center * 2, distance2ArmorCenter * 4.0f, 3, cv::Scalar(255, 0, 0), 0.3));
#else
            img_markers.circles.emplace_back(CircleMarker(frame_header, armor.m_targetshape.m_center, maxDistance2ArmorCenter * 2.0f, 3, cv::Scalar(0, 255, 0), 0.3));
            img_markers.circles.emplace_back(CircleMarker(frame_header, armor.m_targetshape.m_center, minDistance2ArmorCenter * 2.0f, 3, cv::Scalar(0, 255, 0), 0.3));
            img_markers.circles.emplace_back(CircleMarker(frame_header, armor.m_targetshape.m_center, distance2ArmorCenter * 2.0f, 3, cv::Scalar(255, 0, 0), 0.3));
#endif
        }
        std::vector<Lightline> filteredLightlines;
        for (auto iter = lightlines.begin(); iter != lightlines.end(); ++iter) {
            cv::Point2f armorCenter = armor.m_targetshape.m_center;
            double p2p{pointPointDistance(iter->m_center, armorCenter)};
            double p2l{pointLineDistance(iter->m_center, armor.m_center, arrow.m_center)};
            // 判断到装甲板外部灯条的距离
            if (inRange(p2p, minDistance2ArmorCenter, maxDistance2ArmorCenter) == false) {
                continue;
            }
            // 判断到箭头所在直线的距离
            if (p2l > maxDistance2ArrowLine) {
                continue;
            }
            filteredLightlines.emplace_back(*iter);
        }
        if (filteredLightlines.empty()) {
            return false;
        }
        // 取所有符合要求的灯条中面积最大的为中心 R 灯条并设置中心 R
        Lightline target{
                *std::max_element(filteredLightlines.begin(), filteredLightlines.end(),
                                  [](const Lightline &l1, const Lightline &l2) { return l1.m_area < l2.m_area; })};
        center.set(target);
        return true;
    }

    /**
     * @brief 根据已知信息，返回pnp所需的图像标定点
     * @param[in] armor         装甲板信息
     * @param[in] R             中心R
     * @return std::vector<cv::Point2f> 标定点
     */
    std::vector<cv::Point2f> SectorFinder::getCameraPoints(Armor &armor, CenterR &R) {
        std::vector<cv::Point2f> cameraPoints;
        if (armor.m_targetshape.m_children != 4) return cameraPoints;

        std::vector<cv::Point2f> temp{
                cv::minAreaRect(armor.m_targetshape.m_childrenContours[0]).center + armor.m_targetshape.m_roi,
                cv::minAreaRect(armor.m_targetshape.m_childrenContours[1]).center + armor.m_targetshape.m_roi,
                cv::minAreaRect(armor.m_targetshape.m_childrenContours[2]).center + armor.m_targetshape.m_roi,
                cv::minAreaRect(armor.m_targetshape.m_childrenContours[3]).center + armor.m_targetshape.m_roi};

        cv::Point2f UL, UR, DL, DR;
        // 1.用R到装甲板中心距离判断点上下
        std::sort(temp.begin(), temp.end(),
                  [&R](cv::Point2f a, cv::Point2f b) { return pointPointDistance(a, R.m_center) > pointPointDistance(b, R.m_center); });
        UL = temp[0];
        UR = temp[1];
        DL = temp[2];
        DR = temp[3];

        // 2.用 快速排斥实验+跨立实验 判定点左右
        if (areSegmentsIntersecting(UL, DL, UR, DR)) std::swap(UL, UR);

        // 3.向量叉积判断正反
        if (!pointRelativeToLine(armor.m_center, R.m_center, UL)) {
            std::swap(UL, UR);
            std::swap(DL, DR);
        }

#if BAYER_IMAGE
        cameraPoints.emplace_back(UL * 2.0f);            //左上
        cameraPoints.emplace_back(UR * 2.0f);            //右上
        cameraPoints.emplace_back(armor.m_center * 2.0f);//装甲板中心
        cameraPoints.emplace_back(DL * 2.0f);            //左下
        cameraPoints.emplace_back(DR * 2.0f);            //右下
        cameraPoints.emplace_back(R.m_center * 2.0f);    //R中心
#else
        cameraPoints.emplace_back(UL);            //左上
        cameraPoints.emplace_back(UR);            //右上
        cameraPoints.emplace_back(armor.m_center);//装甲板中心
        cameraPoints.emplace_back(DL);            //左下
        cameraPoints.emplace_back(DR);            //右下
        cameraPoints.emplace_back(R.m_center);    //R中心
#endif


        return cameraPoints;
    }

    /**
     * @brief 比较两个灯条是否满足在一个箭头内的条件，是则返回 true，否则为 false
     * @param[in] l1
     * @param[in] l2
     * @return true
     * @return false
     */
    bool SectorFinder::sameArrow(const Lightline &l1, const Lightline &l2) {
        // 判断面积比
        double areaRatio{l1.m_area / l2.m_area};
        if (areaRatio < 1.0f / 5.0f || areaRatio > 5) return false;
        // 判断距离
        double distance{pointPointDistance(l1.m_rotatedRect.center, l2.m_rotatedRect.center)};
        double maxDistance{1.4 * (l1.m_width + l2.m_width)};
        if (distance > maxDistance) return false;

        return true;
    }

    SectorFinder::~SectorFinder() {
#if DETAIL
        // 输出平均处理时间统计
        RCLCPP_INFO(this->get_logger(), "性能统计报告:");
        double total_avg_time = 0.0;
        int total_frames = 0;

        // 计算并输出每个步骤的平均时间
        for (const auto &[step_name, stats]: step_statistics_) {
            if (stats.frame_count > 0) {
                double avg_time = stats.total_time / stats.frame_count;
                total_avg_time += avg_time;
                total_frames = std::max(total_frames, stats.frame_count);
                RCLCPP_INFO(this->get_logger(), "%s - 平均时间: %.2fms (处理帧数: %d)",
                            step_name.c_str(), avg_time, stats.frame_count);
            }
        }

        // 输出总平均时间
        if (total_frames > 0) {
            RCLCPP_INFO(this->get_logger(), "总平均处理时间: %.2fms (总帧数: %d)",
                        total_avg_time, total_frames);
        }
#endif
    }

}// namespace buff_tracker

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(buff_tracker::SectorFinder)