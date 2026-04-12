#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__FINDER__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__FINDER__H
#include "buff_tracker/EllipseFitter.h"
#include "buff_tracker/PnP.h"
#include "buff_tracker/Sector.h"
#include "buff_tracker/Values.h"
#include "buff_tracker/mem_map.h"
#include "ifr_common/defs.h"
#include "rclcpp/rclcpp.hpp"
#include "rm_interface/msg/buff_data.hpp"
#include "rm_referee_interface/msg/game_robot_status.hpp"
#include "rm_referee_interface/msg/game_status.hpp"
#include <cuda_runtime.h>
#include <cv_bridge/cv_bridge.h>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/points_annotation.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <ifr_interface/msg/aim_type.hpp>
#include <ifr_interface/msg/detail/bayer_image__struct.hpp>
#include <ifr_interface/msg/detail/common_ptr__struct.hpp>
#include <ifr_interface/msg/serial_imu_data.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace buff_tracker {
    class SectorFinder : public rclcpp::Node {
    private:
        // 添加性能统计相关结构体
        struct StepTiming {
            double total_time{0.0};
            int frame_count{0};
        };

        // 添加性能统计相关变量
        std::map<std::string, StepTiming> step_statistics_;

    public:
        explicit SectorFinder(const rclcpp::NodeOptions &options);
        virtual ~SectorFinder();

    protected:
        /// 字段
        cv::cuda::Stream stream1{cudaStreamNonBlocking}, stream2{cudaStreamNonBlocking}, stream3{cudaStreamNonBlocking};///< 异步流
        std::unique_ptr<MemMapPool> memMapPool;                                                                         ///< 内存映射池
        std::unique_ptr<Values> values;                                                                                 ///< 所有参数
        std::unique_ptr<PnPSolver> pnp;                                                                                 ///< pnp解算器
        std::unique_ptr<EllipseFitter> ef;                                                                              ///< 椭圆拟合器

    protected:
        ///执行步骤

        void step0_prepare();

        void step1_findContours();

        void step2_goodLight();

        bool step3_goodArrow();

        void step4_setRoi();

        bool step5_detectArmor();

        bool step6_detectCenterR();

        bool step_pnp();

        void step8_setRoi();

        void step9_publish();

        bool step_ef();

        void step_dataCal();

        void step_errorFix();

    protected:
        /// 数据
        bool can_run = true;         ///< 根据自瞄模式选择是否运行
        bool prev_available = false; ///< 是否有可用的上一帧
        unsigned int miss_target = 0;///< 丢失目标帧数

        uint8_t game_state;  ///< 比赛阶段
        Team team;           ///< 队伍信息
        uint16_t remain_time;///< 阶段剩余时间

        std_msgs::msg::Header frame_header;///< 图片时间戳

        cv::Mat cpu_src;         ///< cpu原图
        cv::Mat cpu_hsv;
        //cv::cuda::GpuMat gpu_src;///< gpu原图

        // 三通道图像初始化
        //cv::Mat blue, green, red;

        //cv::cuda::GpuMat c_arrow,d_arrow;       ///< 箭头腐蚀膨胀中间图
        //cv::cuda::GpuMat c_armor,d_armor;       ///< 板子腐蚀膨胀中间图
        //cv::cuda::GpuMat gpu_gray_arrow;///< 箭头灰度图
        //cv::cuda::GpuMat gpu_gray_armor;///< 板子灰度图
        cv::Mat cpu_gray_arrow;///< 箭头灰度图
        cv::Mat cpu_gray_armor;///< 板子灰度图        

        Arrow m_arrow;    // 流水灯条
        Armor m_armor;    // 装甲板
        CenterR m_centerR;// 中心 R

        std::vector<cv::Point2f> cameraPoints;// 能量机关标定点，具体见pnp.h

        cv::Rect prev_armor;  // 上一帧装甲板所在范围
        cv::Rect prev_centerR;// 上一帧中心 R所在范围

        cv::Rect2f m_globalRoi;// 全局 roi，用来圈定识别的范围，加快处理速度
        cv::Rect2f m_armorRoi; // 装甲板 roi
        cv::Rect2f m_centerRoi;// 中心 R roi

        cv::Mat m_imageArrow; // 检测箭头用的二值化图片
        cv::Mat m_imageArmor; // 检测装甲板边框用的二值化图片
        cv::Mat m_imageCenter;// 检测中心 R 用的二值化图片
        cv::Mat m_localMask;  // 局部 roi 的掩码

        std::vector<Lightline> lightlines;           ///< 流水灯条集合
        std::vector<std::vector<cv::Point>> contours;///< 所有轮廓

    protected:
        /// 消息
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_arrow;///< 流水灯条阈值图
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_armor;///< 装甲板阈值图

        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_img_preview;///< 预览图

        foxglove_msgs::msg::ImageAnnotations img_markers;///< 所有图像标记
        rclcpp::Publisher<decltype(img_markers)>::SharedPtr pub_img_markers;//decltype(变量)：C++11 关键字，自动提取变量的类型。

        visualization_msgs::msg::MarkerArray msg_markers;///< 所有三维标记
        rclcpp::Publisher<decltype(msg_markers)>::SharedPtr pub_markers;

        visualization_msgs::msg::Marker meshMarker; ///< 单个装甲板标记
        visualization_msgs::msg::Marker arrowMarker;///< 单个装甲板标记
        visualization_msgs::msg::Marker lineMarker; ///< 原点到能量机关连线

        rm_interface::msg::BuffData buffdata;///< 能量机关信息
        rclcpp::Publisher<decltype(buffdata)>::SharedPtr pub_buffdata;

#if BAYER_IMAGE
        rclcpp::Subscription<ifr_interface::msg::BayerImage>::SharedPtr sub_src;///< 订阅Bayer图片消息
        ifr_interface::msg::CommonPtr msg_back_ptr;                             ///< 图像指针返还
        rclcpp::Publisher<decltype(msg_back_ptr)>::SharedPtr pub_back_ptr;
#else
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_src;///< 订阅Image图片消息
#endif

        rclcpp::Subscription<ifr_interface::msg::SerialImuData>::SharedPtr sub_time;///< 时间输入
        rclcpp::Subscription<ifr_interface::msg::Team>::SharedPtr sub_team;         ///< 队伍输入
        rclcpp::Subscription<ifr_interface::msg::AimType>::SharedPtr sub_aim_type;  ///< 自瞄类型
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cam_info; ///< 相机参数

    protected:
        ///其它
        /**
            * 图像输入
            * @param img_msg 图像数据
            */
        void rgb_input(const sensor_msgs::msg::Image::SharedPtr img_msg);

        void bayer_input(const ifr_interface::msg::BayerImage::SharedPtr msg);

        void recognise(std_msgs::msg::Header &frame_header, cv::Mat &cpu_src, uint64_t data = 0);

        void extract_mask(const cv::Mat& h,   const cv::Mat& s,   const cv::Mat& v,
            cv::Mat& dst_arrow, cv::Mat& dst_armor,
            int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
            int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m)

        bool findArmor(Armor &armor, const std::vector<TargetShape> &TargetShapes);

        bool findArmorTargetShape(const cv::Mat &binary, TargetShape &TargetCross, std::vector<std::string> &err,
                                  const cv::Rect2f &globalRoi, const cv::Rect2f &localRoi);

        bool findCenterLightlines(const cv::Mat &binary, std::vector<Lightline> &lightlines,
                                  const cv::Rect2f &globalRoi, const cv::Rect2f &localRoi);

        bool findCenterR(CenterR &center, const std::vector<Lightline> &lightlines, const Arrow &arrowPtr,
                         const Armor &armor);

        std::vector<cv::Point2f> getCameraPoints(Armor &armor, CenterR &R);

        static bool sameArrow(const Lightline &l1, const Lightline &l2);

        bool ErrorMessage(Center_Status msg);
    };

}// namespace buff_tracker

#endif// IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__FINDER__H