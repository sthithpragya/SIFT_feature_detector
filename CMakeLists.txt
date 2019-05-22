cmake_minimum_required(VERSION 2.8.3)
project( tracking )
find_package( OpenCV REQUIRED )
add_executable(tracking_centroid main/tracking_centroid.cpp )
add_executable(tracking_wrt_frame1 main/tracking_wrt_frame1.cpp )
target_link_libraries( tracking_centroid ${OpenCV_LIBS} )
target_link_libraries( tracking_wrt_frame1 ${OpenCV_LIBS} )

