#ifndef IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__CUDAS__H
#define IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__CUDAS__H
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace buff_tracker::cudas {

    /**
     * @brief 从RGB三通道图像中通过HSV空间获取Armor和Arrow掩码（更鲁棒）
     * @param r     R通道图像
     * @param g     G通道图像
     * @param b     B通道图像
     * @param dst_arrow  输出 Arrow 掩码（0/255）cv::cuda::PtrStepSz<uint8_t> 
     * @param dst_armor  输出 Armor 掩码（0/255）cv::cuda::PtrStepSz<uint8_t> 
     * @param h_low_a    Arrow 的 H 最小值（0–360）
     * @param h_high_a   Arrow 的 H 最大值
     * @param s_low_a    Arrow 的 S 最小值（0–255）
     * @param s_high_a   Arrow 的 S 最大值
     * @param v_low_a    Arrow 的 V 最小值（0–255）
     * @param v_high_a   Arrow 的 V 最大值
     * @param h_low_m    Armor 的 H 最小值
     * @param h_high_m   Armor 的 H 最大值
     * @param s_low_m    Armor 的 S 最小值
     * @param s_high_m   Armor 的 S 最大值
     * @param v_low_m    Armor 的 V 最小值
     * @param v_high_m   Armor 的 V 最大值
     * @param stream     CUDA 流
     */
    void getColorFromRGB_hsv(
        const cv::cuda::GpuMat& r,   const cv::cuda::GpuMat& g,   const cv::cuda::GpuMat& b,
        cv::cuda::GpuMat& dst_arrow, cv::cuda::GpuMat& dst_armor,
        int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
        int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    /**
     * @brief 从 Bayer‑RG 格式单通道图像中通过 HSV 空间获取 Armor/Arrow 掩码（更鲁棒）
     * @param src        原始 Bayer‑RG 单通道图
     * @param dst_arrow  输出 Arrow 掩码
     * @param dst_armor  输出 Armor 掩码
     * @param h_low_a    Arrow H 下界（0–360）
     * @param h_high_a   Arrow H 上界
     * @param s_low_a    Arrow S 下界（0–255）
     * @param s_high_a   Arrow S 上界
     * @param v_low_a    Arrow V 下界（0–255）
     * @param v_high_a   Arrow V 上界
     * @param h_low_m    Armor H 下界
     * @param h_high_m   Armor H 上界
     * @param s_low_m    Armor S 下界
     * @param s_high_m   Armor S 上界
     * @param v_low_m    Armor V 下界
     * @param v_high_m   Armor V 上界
     * @param stream     CUDA 流
     */
    void getColorFromBayerRG_hsv(
        const cv::cuda::GpuMat& src,
        cv::cuda::GpuMat& dst_arrow,
        cv::cuda::GpuMat& dst_armor,
        int h_low_a, int h_high_a, int s_low_a, int s_high_a, int v_low_a, int v_high_a,
        int h_low_m, int h_high_m, int s_low_m, int s_high_m, int v_low_m, int v_high_m,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

}// namespace buff_tracker::cudas

#endif//IFR_ROS2_CV__PACKAGE_RM_BUFF_TRACKER__CUDAS__H