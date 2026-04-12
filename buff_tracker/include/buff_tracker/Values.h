#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__VALUES__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__VALUES__H
#include <cstdint>
#include <functional>
#include <opencv2/opencv.hpp>
#include <rcl_interfaces/msg/detail/floating_point_range__struct.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/parameter_value.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
namespace buff_tracker {
    //该类是 ROS2 参数动态管理工具，核心作用是校验并执行注册参数的修改，拦截未注册的val_前缀参数并返回错误
    class Values {
        static constexpr const char prefix[] = "val_";
        rclcpp::Node *node = nullptr;

        std::unordered_map<std::string, std::function<void(const rclcpp::Parameter &)>> all_params;///< 参数地址保存
        ///参数更改
        rcl_interfaces::msg::SetParametersResult callback(const std::vector<rclcpp::Parameter> &params) {
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true;
            for (const auto &param: params) {
                const auto &name = param.get_name();
                if (all_params.count(name)) {
                    all_params[name](param);
                } else if (!name.compare(0, sizeof(prefix) - 1, prefix)) {
                    result.successful = false;
                    result.reason = "Not Found: " + name;
                }
            }
            return result;
        }
        /**
        声明一个变量
        @param type 类型名称
        @param name 变量名称
        @param val 默认值
        @param min_v 最小值
        @param max_v 最大值
        @param desc 描述
        @param ptr 修改地址 (小于0则不修改)
        */
        template<class T>
        T declare(std::string type, std::string name, T val, T min_v, T max_v, std::string desc, int64_t ptr = -1) {
            name = prefix + name;
            rcl_interfaces::msg::ParameterDescriptor descriptor;
            descriptor.name = name;
            descriptor.description = desc;
            if constexpr (std::is_same_v<std::remove_cv_t<T>, bool>) {
            } else if constexpr (std::is_integral_v<T>) {
                rcl_interfaces::msg::IntegerRange range;
                range.from_value = min_v, range.to_value = max_v;// range.step = 1;
                descriptor.integer_range.emplace_back(range);
            } else if constexpr (std::is_floating_point_v<T>) {
                rcl_interfaces::msg::FloatingPointRange range;
                range.from_value = min_v, range.to_value = max_v;//, range.step = std::numeric_limits<T>::epsilon();
                descriptor.floating_point_range.emplace_back(range);
            }
            if (ptr >= 0) {
                all_params[name] = [this, ptr](const rclcpp::Parameter &p) {
                    T *target = reinterpret_cast<T *>((reinterpret_cast<uint8_t *>(this) + ptr));
                    if constexpr (std::is_same_v<std::remove_cv_t<T>, bool>)
                        *target = p.as_bool();
                    else if constexpr (std::is_integral_v<T>)
                        *target = static_cast<T>(p.as_int());
                    else if constexpr (std::is_floating_point_v<T>)
                        *target = static_cast<T>(p.as_double());
                };
            } else
                descriptor.read_only = true;

            auto new_val = node->declare_parameter(name, val, descriptor);
            RCLCPP_DEBUG_STREAM(
                    node->get_logger(),
                    "declare var" << (ptr >= 0 ? "" : "(ro)") << ": "
                                  << type << " " << name << " = " << new_val << "(" << val << ") // " << desc << "min:" << min_v << "max:" << max_v);
            return new_val;
        }

    public:
        explicit Values(rclcpp::Node *node) : node(node) {
            params_callback_handle_ = node->add_on_set_parameters_callback(
                    std::bind(&Values::callback, this, std::placeholders::_1));
        }
#define VALUES_PREFIX static constexpr const
#define DECLARE_VALUES_R(type, name, val, min, max, desc) \
public:                                                   \
    const type name = declare<std::remove_cv_t<type>>(#type, #name, val, min, max, desc)
#define DECLARE_VALUES_M(type, name, val, min, max, desc) \
public:                                                   \
    type name = declare<std::remove_cv_t<type>>(#type, #name, val, min, max, desc, reinterpret_cast<intptr_t>(&reinterpret_cast<Values *>(0)->name))
        DECLARE_VALUES_M(float, DELTA_TIME, 0, 0, 3, "预测时间延时");
        //buff
        DECLARE_VALUES_M(int, RED_CUDA_ARMOR_HL, 210, 0, 360, "红方装甲板的色调下限");
        DECLARE_VALUES_M(int, RED_CUDA_ARMOR_HH, 210, 0, 360, "红方装甲板的色调上限");
        DECLARE_VALUES_M(int, RED_CUDA_ARMOR_SL, 210, 0, 255, "红方装甲板的饱和度下限");
        DECLARE_VALUES_M(int, RED_CUDA_ARMOR_SH, 210, 0, 255, "红方装甲板的饱和度上限");
        DECLARE_VALUES_M(int, RED_CUDA_ARMOR_VL, 210, 0, 255, "红方装甲板的亮度下限");
        DECLARE_VALUES_M(int, RED_CUDA_ARMOR_VH, 210, 0, 255, "红方装甲板的亮度上限");

        DECLARE_VALUES_M(int, RED_CUDA_ARROW_HL, 210, 0, 360, "红方流水灯的色调下限");
        DECLARE_VALUES_M(int, RED_CUDA_ARROW_HH, 210, 0, 360, "红方流水灯的色调上限");
        DECLARE_VALUES_M(int, RED_CUDA_ARROW_SL, 210, 0, 255, "红方流水灯的饱和度下限");
        DECLARE_VALUES_M(int, RED_CUDA_ARROW_SH, 210, 0, 255, "红方流水灯的饱和度上限");
        DECLARE_VALUES_M(int, RED_CUDA_ARROW_VL, 210, 0, 255, "红方流水灯的亮度下限");
        DECLARE_VALUES_M(int, RED_CUDA_ARROW_VH, 210, 0, 255, "红方流水灯的亮度上限");
        //armor_finder
        DECLARE_VALUES_M(int, BLUE_CUDA_ARMOR_HL, 210, 0, 360, "蓝方装甲板的色调下限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARMOR_HH, 210, 0, 360, "蓝方装甲板的色调上限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARMOR_SL, 210, 0, 255, "蓝方装甲板的饱和度下限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARMOR_SH, 210, 0, 255, "蓝方装甲板的饱和度上限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARMOR_VL, 210, 0, 255, "蓝方装甲板的亮度下限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARMOR_VH, 210, 0, 255, "蓝方装甲板的亮度上限");

        DECLARE_VALUES_M(int, BLUE_CUDA_ARROW_HL, 210, 0, 360, "蓝方流水灯的色调下限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARROW_HH, 210, 0, 360, "蓝方流水灯的色调上限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARROW_SL, 210, 0, 255, "蓝方流水灯的饱和度下限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARROW_SH, 210, 0, 255, "蓝方流水灯的饱和度上限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARROW_VL, 210, 0, 255, "蓝方流水灯的亮度下限");
        DECLARE_VALUES_M(int, BLUE_CUDA_ARROW_VH, 210, 0, 255, "蓝方流水灯的亮度上限");

        DECLARE_VALUES_M(float, LOCAL_ROI_DISTANCE_RATIO, 1.2F, 0, 10, "两个ROI的中心点与流水灯条的距离比");
        DECLARE_VALUES_M(float, MAX_ARROW_LIGHTLINE_ASPECT_RATIO, 3.0F, 0, 10, "流水灯箭头的最大长宽比");
        DECLARE_VALUES_M(double, MAX_ARROW_LIGHTLINE_AREA, 1500.0F, 0, 10000, "流水灯箭头的最大面积");
        DECLARE_VALUES_M(double, MIN_ARROW_LIGHTLINE_AREA, 30.0F, 0, 1000, "流水灯箭头的最小面积");
        DECLARE_VALUES_M(float, LOCAL_ROI_LENTH, 200.0F, 0, 1000, "两个ROI的方框边长");
        DECLARE_VALUES_M(float, GLOBAL_ROI_LENGTH_RATIO, 2.0, 0, 10, "能量机关ROI的方框边长与能量机关直径比");

        DECLARE_VALUES_M(double, MIN_ARROW_ASPECT_RATIO, 2.0, 0, 10, "最小流水灯长宽比");
        DECLARE_VALUES_M(double, MAX_ARROW_ASPECT_RATIO, 12.0, 1, 20, "最大流水灯长宽比");

        VALUES_PREFIX double POWER_RUNE_RADIUS = 700.0f;       // 能量机关半径
        VALUES_PREFIX double ARMOR_TARGETSHAPE_RADIUS = 288.0f;// 十字标半径

        DECLARE_VALUES_M(double, MAX_ARMOR_TARGETSHAPE_CONTOUR_AREA, 15000, 10000, 100000, "最大击打板灯条轮廓面积");
        DECLARE_VALUES_M(double, MAX_ARMOR_TARGETSHAPE_ASPECT_RATIO, 6, 1, 10, "最大击打板灯长宽比");
        DECLARE_VALUES_M(double, MIN_ARMOR_TARGETSHAPE_CONTOUR_AREA, 5000, 1, 20000, "最小击打板灯条轮廓面积");
        DECLARE_VALUES_M(double, MIN_ARMOR_TARGETSHAPE_ASPECT_RATIO, 1.4, 1, 10, "最小击打板灯长宽比");

        DECLARE_VALUES_M(int, BUFF_MODE, 1, 0, 2, "能量机关模式(0:自动 1:小符 2:大符)");

        DECLARE_VALUES_M(int, MAX_ARMOR_TARGETSHAPE_CHILDREN_NUM, 8, 1, 20, "最大十字标子轮廓数量");
        DECLARE_VALUES_M(int, MIN_ARMOR_TARGETSHAPE_CHILDREN_NUM, 2, 1, 20, "最小十字标子轮廓数量");

        DECLARE_VALUES_M(int, MIN_ARROW_LIGHTLINE_NUM, 4, 1, 20, "最小流水灯箭头数量");
        DECLARE_VALUES_M(int, MAX_ARROW_LIGHTLINE_NUM, 13, 1, 20, "最大流水灯箭头数量");
        DECLARE_VALUES_M(float, MAX_ARROW_AREA, 5000, 1, 40000, "最大流水灯面积");

        DECLARE_VALUES_M(double, MIN_CENTER_AREA, 100, 1, 10000, "中心R的最小面积");
        DECLARE_VALUES_M(double, MAX_CENTER_AREA, 1000, 1, 10000, "中心R的最大面积");

        DECLARE_VALUES_M(float, MAX_CENTER_ASPECT_RATIO, 2, 0, 10, "中心R的最大长宽比");

        // EF
        DECLARE_VALUES_M(int, MIN_NUM2FIT, 30, 0, 100, "最小拟合点数");
        DECLARE_VALUES_M(float, RANSAC_CONFIDENCE, 0.99, 0.5, 1, "RANSAC置信度");
        DECLARE_VALUES_M(float, INLIER_THRESHOLD, 5, 1, 50, "内点距离阈值(像素)");
        DECLARE_VALUES_M(int, MAX_ITERATIONS, 10, 3, 20, "RANSAC最大迭代次数");
        DECLARE_VALUES_M(int, MAX_MAPPING_HISTORY, 1, 1, 10, "最大历史记录数");
        DECLARE_VALUES_M(int, MAX_POINTS, 100, 20, 200, "最大存储点数");
        DECLARE_VALUES_M(float, MIN_INLIER_RATIO, 0.5, 0, 3, "最小内点比例");
        DECLARE_VALUES_M(float, MIN_POINT_DISTENCE, 15, 0, 200, "最小点间距阈值");
        DECLARE_VALUES_M(int, MIN_SAMPLES, 6, 5, 30, "最小拟合样本数");

#undef VALUES_PREFIX
#undef DECLARE_VALUES_R
#undef DECLARE_VALUES_M
    private:
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;///< 监听回调句柄
    };

}// namespace buff_tracker
#endif// IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__VALUES__H
