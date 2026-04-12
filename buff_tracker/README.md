# buff_tracker
> 能量机关追踪

识别图片中扇叶并发布能力机关信息


## Publisher

 - `/image_bayer/return_ptr`: 返回Bayer图像指针 `ifr_interface/msg/CommonPtr`  
 - `/rm/finder/armors`: 单帧发现的装甲板 `ifr_interface/msg/RmArmors`  
 - `/rm/finder/debug/armor`: 调试用输出装甲板位置 `geometry_msgs/msg/Pose`  
 - `/rm/finder/debug/gray`: 调试用输出灰度图 `sensor_msgs/msg/Image`  
 - `/rm/finder/debug/light`: 调试用输出灯条图 `sensor_msgs/msg/Image`  
 - `/rm/finder/debug/target`: 调试用输出目标图 `sensor_msgs/msg/Image`  
 - `/rm/finder/markers`: 调试用输出标记 `visualization_msgs/msg/MarkerArray`  

## Subscriber

 - `/image_bayer/image`: Bayer图像输入 `ifr_interface/msg/BayerImage`  
 - `/rm/self_team`: 监听自身队伍 `ifr_interface/msg/Team`  
 - `/image_bayer/camera_info`: 监听相机参数(仅一次) `sensor_msgs/msg/CameraInfo`  