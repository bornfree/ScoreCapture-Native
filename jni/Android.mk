LOCAL_PATH := $(call my-dir)

OPENCV_INSTALL_MODULES:=on
OPENCV_CAMERA_MODULES:=off

include /home/harsha/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk

#include $(CLEAR_VARS)

LOCAL_MODULE    := CameraOMRNative
LOCAL_SRC_FILES := CameraOMRNative.cpp
LOCAL_LDLIBS +=  -llog

include $(BUILD_SHARED_LIBRARY)
