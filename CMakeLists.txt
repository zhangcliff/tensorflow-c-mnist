cmake_minimum_required (VERSION 2.8.8)
project (tf_example)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")


link_directories(./lib)



include_directories(
   /home/zwx/tensorflow
   /home/zwx/tensorflow/bazel-genfiles
   /home/zwx/tensorflow/bazel-bin/tensorflow
   /home/zwx/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
   /home/zwx/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include
   /usr/local/include/eigen3
   ) 
add_executable(tf_test  tf.cpp) 
target_link_libraries(tf_test tensorflow_cc tensorflow_framework)

