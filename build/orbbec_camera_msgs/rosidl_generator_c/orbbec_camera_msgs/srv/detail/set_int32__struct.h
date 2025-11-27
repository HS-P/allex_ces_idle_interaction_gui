// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from orbbec_camera_msgs:srv/SetInt32.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "orbbec_camera_msgs/srv/set_int32.h"


#ifndef ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_INT32__STRUCT_H_
#define ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_INT32__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetInt32 in the package orbbec_camera_msgs.
typedef struct orbbec_camera_msgs__srv__SetInt32_Request
{
  int32_t data;
} orbbec_camera_msgs__srv__SetInt32_Request;

// Struct for a sequence of orbbec_camera_msgs__srv__SetInt32_Request.
typedef struct orbbec_camera_msgs__srv__SetInt32_Request__Sequence
{
  orbbec_camera_msgs__srv__SetInt32_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} orbbec_camera_msgs__srv__SetInt32_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SetInt32 in the package orbbec_camera_msgs.
typedef struct orbbec_camera_msgs__srv__SetInt32_Response
{
  bool success;
  rosidl_runtime_c__String message;
} orbbec_camera_msgs__srv__SetInt32_Response;

// Struct for a sequence of orbbec_camera_msgs__srv__SetInt32_Response.
typedef struct orbbec_camera_msgs__srv__SetInt32_Response__Sequence
{
  orbbec_camera_msgs__srv__SetInt32_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} orbbec_camera_msgs__srv__SetInt32_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  orbbec_camera_msgs__srv__SetInt32_Event__request__MAX_SIZE = 1
};
// response
enum
{
  orbbec_camera_msgs__srv__SetInt32_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/SetInt32 in the package orbbec_camera_msgs.
typedef struct orbbec_camera_msgs__srv__SetInt32_Event
{
  service_msgs__msg__ServiceEventInfo info;
  orbbec_camera_msgs__srv__SetInt32_Request__Sequence request;
  orbbec_camera_msgs__srv__SetInt32_Response__Sequence response;
} orbbec_camera_msgs__srv__SetInt32_Event;

// Struct for a sequence of orbbec_camera_msgs__srv__SetInt32_Event.
typedef struct orbbec_camera_msgs__srv__SetInt32_Event__Sequence
{
  orbbec_camera_msgs__srv__SetInt32_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} orbbec_camera_msgs__srv__SetInt32_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_INT32__STRUCT_H_
