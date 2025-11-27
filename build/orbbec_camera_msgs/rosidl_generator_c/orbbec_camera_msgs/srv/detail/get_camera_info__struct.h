// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from orbbec_camera_msgs:srv/GetCameraInfo.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "orbbec_camera_msgs/srv/get_camera_info.h"


#ifndef ORBBEC_CAMERA_MSGS__SRV__DETAIL__GET_CAMERA_INFO__STRUCT_H_
#define ORBBEC_CAMERA_MSGS__SRV__DETAIL__GET_CAMERA_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetCameraInfo in the package orbbec_camera_msgs.
typedef struct orbbec_camera_msgs__srv__GetCameraInfo_Request
{
  uint8_t structure_needs_at_least_one_member;
} orbbec_camera_msgs__srv__GetCameraInfo_Request;

// Struct for a sequence of orbbec_camera_msgs__srv__GetCameraInfo_Request.
typedef struct orbbec_camera_msgs__srv__GetCameraInfo_Request__Sequence
{
  orbbec_camera_msgs__srv__GetCameraInfo_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} orbbec_camera_msgs__srv__GetCameraInfo_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "sensor_msgs/msg/detail/camera_info__struct.h"

/// Struct defined in srv/GetCameraInfo in the package orbbec_camera_msgs.
typedef struct orbbec_camera_msgs__srv__GetCameraInfo_Response
{
  sensor_msgs__msg__CameraInfo info;
} orbbec_camera_msgs__srv__GetCameraInfo_Response;

// Struct for a sequence of orbbec_camera_msgs__srv__GetCameraInfo_Response.
typedef struct orbbec_camera_msgs__srv__GetCameraInfo_Response__Sequence
{
  orbbec_camera_msgs__srv__GetCameraInfo_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} orbbec_camera_msgs__srv__GetCameraInfo_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  orbbec_camera_msgs__srv__GetCameraInfo_Event__request__MAX_SIZE = 1
};
// response
enum
{
  orbbec_camera_msgs__srv__GetCameraInfo_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/GetCameraInfo in the package orbbec_camera_msgs.
typedef struct orbbec_camera_msgs__srv__GetCameraInfo_Event
{
  service_msgs__msg__ServiceEventInfo info;
  orbbec_camera_msgs__srv__GetCameraInfo_Request__Sequence request;
  orbbec_camera_msgs__srv__GetCameraInfo_Response__Sequence response;
} orbbec_camera_msgs__srv__GetCameraInfo_Event;

// Struct for a sequence of orbbec_camera_msgs__srv__GetCameraInfo_Event.
typedef struct orbbec_camera_msgs__srv__GetCameraInfo_Event__Sequence
{
  orbbec_camera_msgs__srv__GetCameraInfo_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} orbbec_camera_msgs__srv__GetCameraInfo_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ORBBEC_CAMERA_MSGS__SRV__DETAIL__GET_CAMERA_INFO__STRUCT_H_
