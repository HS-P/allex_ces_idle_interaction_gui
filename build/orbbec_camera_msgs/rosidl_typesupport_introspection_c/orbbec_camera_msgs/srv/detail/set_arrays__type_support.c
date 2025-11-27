// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from orbbec_camera_msgs:srv/SetArrays.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "orbbec_camera_msgs/srv/detail/set_arrays__rosidl_typesupport_introspection_c.h"
#include "orbbec_camera_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "orbbec_camera_msgs/srv/detail/set_arrays__functions.h"
#include "orbbec_camera_msgs/srv/detail/set_arrays__struct.h"


// Include directives for member types
// Member `data_param`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  orbbec_camera_msgs__srv__SetArrays_Request__init(message_memory);
}

void orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_fini_function(void * message_memory)
{
  orbbec_camera_msgs__srv__SetArrays_Request__fini(message_memory);
}

size_t orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__size_function__SetArrays_Request__data_param(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Request__data_param(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__get_function__SetArrays_Request__data_param(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__fetch_function__SetArrays_Request__data_param(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Request__data_param(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__assign_function__SetArrays_Request__data_param(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__get_function__SetArrays_Request__data_param(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__resize_function__SetArrays_Request__data_param(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_member_array[2] = {
  {
    "enable",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Request, enable),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "data_param",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Request, data_param),  // bytes offset in struct
    NULL,  // default value
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__size_function__SetArrays_Request__data_param,  // size() function pointer
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Request__data_param,  // get_const(index) function pointer
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__get_function__SetArrays_Request__data_param,  // get(index) function pointer
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__fetch_function__SetArrays_Request__data_param,  // fetch(index, &value) function pointer
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__assign_function__SetArrays_Request__data_param,  // assign(index, value) function pointer
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__resize_function__SetArrays_Request__data_param  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_members = {
  "orbbec_camera_msgs__srv",  // message namespace
  "SetArrays_Request",  // message name
  2,  // number of fields
  sizeof(orbbec_camera_msgs__srv__SetArrays_Request),
  false,  // has_any_key_member_
  orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_member_array,  // message members
  orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_type_support_handle = {
  0,
  &orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_members,
  get_message_typesupport_handle_function,
  &orbbec_camera_msgs__srv__SetArrays_Request__get_type_hash,
  &orbbec_camera_msgs__srv__SetArrays_Request__get_type_description,
  &orbbec_camera_msgs__srv__SetArrays_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_orbbec_camera_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Request)() {
  if (!orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_type_support_handle.typesupport_identifier) {
    orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__rosidl_typesupport_introspection_c.h"
// already included above
// #include "orbbec_camera_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__functions.h"
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__struct.h"


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  orbbec_camera_msgs__srv__SetArrays_Response__init(message_memory);
}

void orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_fini_function(void * message_memory)
{
  orbbec_camera_msgs__srv__SetArrays_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_member_array[2] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_members = {
  "orbbec_camera_msgs__srv",  // message namespace
  "SetArrays_Response",  // message name
  2,  // number of fields
  sizeof(orbbec_camera_msgs__srv__SetArrays_Response),
  false,  // has_any_key_member_
  orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_member_array,  // message members
  orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle = {
  0,
  &orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_members,
  get_message_typesupport_handle_function,
  &orbbec_camera_msgs__srv__SetArrays_Response__get_type_hash,
  &orbbec_camera_msgs__srv__SetArrays_Response__get_type_description,
  &orbbec_camera_msgs__srv__SetArrays_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_orbbec_camera_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Response)() {
  if (!orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle.typesupport_identifier) {
    orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__rosidl_typesupport_introspection_c.h"
// already included above
// #include "orbbec_camera_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__functions.h"
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "orbbec_camera_msgs/srv/set_arrays.h"
// Member `request`
// Member `response`
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  orbbec_camera_msgs__srv__SetArrays_Event__init(message_memory);
}

void orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_fini_function(void * message_memory)
{
  orbbec_camera_msgs__srv__SetArrays_Event__fini(message_memory);
}

size_t orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__size_function__SetArrays_Event__request(
  const void * untyped_member)
{
  const orbbec_camera_msgs__srv__SetArrays_Request__Sequence * member =
    (const orbbec_camera_msgs__srv__SetArrays_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Event__request(
  const void * untyped_member, size_t index)
{
  const orbbec_camera_msgs__srv__SetArrays_Request__Sequence * member =
    (const orbbec_camera_msgs__srv__SetArrays_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_function__SetArrays_Event__request(
  void * untyped_member, size_t index)
{
  orbbec_camera_msgs__srv__SetArrays_Request__Sequence * member =
    (orbbec_camera_msgs__srv__SetArrays_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__fetch_function__SetArrays_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const orbbec_camera_msgs__srv__SetArrays_Request * item =
    ((const orbbec_camera_msgs__srv__SetArrays_Request *)
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Event__request(untyped_member, index));
  orbbec_camera_msgs__srv__SetArrays_Request * value =
    (orbbec_camera_msgs__srv__SetArrays_Request *)(untyped_value);
  *value = *item;
}

void orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__assign_function__SetArrays_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  orbbec_camera_msgs__srv__SetArrays_Request * item =
    ((orbbec_camera_msgs__srv__SetArrays_Request *)
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_function__SetArrays_Event__request(untyped_member, index));
  const orbbec_camera_msgs__srv__SetArrays_Request * value =
    (const orbbec_camera_msgs__srv__SetArrays_Request *)(untyped_value);
  *item = *value;
}

bool orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__resize_function__SetArrays_Event__request(
  void * untyped_member, size_t size)
{
  orbbec_camera_msgs__srv__SetArrays_Request__Sequence * member =
    (orbbec_camera_msgs__srv__SetArrays_Request__Sequence *)(untyped_member);
  orbbec_camera_msgs__srv__SetArrays_Request__Sequence__fini(member);
  return orbbec_camera_msgs__srv__SetArrays_Request__Sequence__init(member, size);
}

size_t orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__size_function__SetArrays_Event__response(
  const void * untyped_member)
{
  const orbbec_camera_msgs__srv__SetArrays_Response__Sequence * member =
    (const orbbec_camera_msgs__srv__SetArrays_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Event__response(
  const void * untyped_member, size_t index)
{
  const orbbec_camera_msgs__srv__SetArrays_Response__Sequence * member =
    (const orbbec_camera_msgs__srv__SetArrays_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_function__SetArrays_Event__response(
  void * untyped_member, size_t index)
{
  orbbec_camera_msgs__srv__SetArrays_Response__Sequence * member =
    (orbbec_camera_msgs__srv__SetArrays_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__fetch_function__SetArrays_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const orbbec_camera_msgs__srv__SetArrays_Response * item =
    ((const orbbec_camera_msgs__srv__SetArrays_Response *)
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Event__response(untyped_member, index));
  orbbec_camera_msgs__srv__SetArrays_Response * value =
    (orbbec_camera_msgs__srv__SetArrays_Response *)(untyped_value);
  *value = *item;
}

void orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__assign_function__SetArrays_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  orbbec_camera_msgs__srv__SetArrays_Response * item =
    ((orbbec_camera_msgs__srv__SetArrays_Response *)
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_function__SetArrays_Event__response(untyped_member, index));
  const orbbec_camera_msgs__srv__SetArrays_Response * value =
    (const orbbec_camera_msgs__srv__SetArrays_Response *)(untyped_value);
  *item = *value;
}

bool orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__resize_function__SetArrays_Event__response(
  void * untyped_member, size_t size)
{
  orbbec_camera_msgs__srv__SetArrays_Response__Sequence * member =
    (orbbec_camera_msgs__srv__SetArrays_Response__Sequence *)(untyped_member);
  orbbec_camera_msgs__srv__SetArrays_Response__Sequence__fini(member);
  return orbbec_camera_msgs__srv__SetArrays_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Event, request),  // bytes offset in struct
    NULL,  // default value
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__size_function__SetArrays_Event__request,  // size() function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Event__request,  // get_const(index) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_function__SetArrays_Event__request,  // get(index) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__fetch_function__SetArrays_Event__request,  // fetch(index, &value) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__assign_function__SetArrays_Event__request,  // assign(index, value) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__resize_function__SetArrays_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(orbbec_camera_msgs__srv__SetArrays_Event, response),  // bytes offset in struct
    NULL,  // default value
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__size_function__SetArrays_Event__response,  // size() function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_const_function__SetArrays_Event__response,  // get_const(index) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__get_function__SetArrays_Event__response,  // get(index) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__fetch_function__SetArrays_Event__response,  // fetch(index, &value) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__assign_function__SetArrays_Event__response,  // assign(index, value) function pointer
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__resize_function__SetArrays_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_members = {
  "orbbec_camera_msgs__srv",  // message namespace
  "SetArrays_Event",  // message name
  3,  // number of fields
  sizeof(orbbec_camera_msgs__srv__SetArrays_Event),
  false,  // has_any_key_member_
  orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_member_array,  // message members
  orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_type_support_handle = {
  0,
  &orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_members,
  get_message_typesupport_handle_function,
  &orbbec_camera_msgs__srv__SetArrays_Event__get_type_hash,
  &orbbec_camera_msgs__srv__SetArrays_Event__get_type_description,
  &orbbec_camera_msgs__srv__SetArrays_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_orbbec_camera_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Event)() {
  orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Request)();
  orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Response)();
  if (!orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_type_support_handle.typesupport_identifier) {
    orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "orbbec_camera_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_arrays__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_members = {
  "orbbec_camera_msgs__srv",  // service namespace
  "SetArrays",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_Request_message_type_support_handle,
  NULL,  // response message
  // orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle
  NULL  // event_message
  // orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle
};


static rosidl_service_type_support_t orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_type_support_handle = {
  0,
  &orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_members,
  get_service_typesupport_handle_function,
  &orbbec_camera_msgs__srv__SetArrays_Request__rosidl_typesupport_introspection_c__SetArrays_Request_message_type_support_handle,
  &orbbec_camera_msgs__srv__SetArrays_Response__rosidl_typesupport_introspection_c__SetArrays_Response_message_type_support_handle,
  &orbbec_camera_msgs__srv__SetArrays_Event__rosidl_typesupport_introspection_c__SetArrays_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    orbbec_camera_msgs,
    srv,
    SetArrays
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    orbbec_camera_msgs,
    srv,
    SetArrays
  ),
  &orbbec_camera_msgs__srv__SetArrays__get_type_hash,
  &orbbec_camera_msgs__srv__SetArrays__get_type_description,
  &orbbec_camera_msgs__srv__SetArrays__get_type_description_sources,
};

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_orbbec_camera_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays)(void) {
  if (!orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_type_support_handle.typesupport_identifier) {
    orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, orbbec_camera_msgs, srv, SetArrays_Event)()->data;
  }

  return &orbbec_camera_msgs__srv__detail__set_arrays__rosidl_typesupport_introspection_c__SetArrays_service_type_support_handle;
}
