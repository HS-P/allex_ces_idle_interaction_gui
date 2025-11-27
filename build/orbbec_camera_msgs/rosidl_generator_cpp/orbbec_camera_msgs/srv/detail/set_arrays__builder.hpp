// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from orbbec_camera_msgs:srv/SetArrays.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "orbbec_camera_msgs/srv/set_arrays.hpp"


#ifndef ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_ARRAYS__BUILDER_HPP_
#define ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_ARRAYS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "orbbec_camera_msgs/srv/detail/set_arrays__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace orbbec_camera_msgs
{

namespace srv
{

namespace builder
{

class Init_SetArrays_Request_data_param
{
public:
  explicit Init_SetArrays_Request_data_param(::orbbec_camera_msgs::srv::SetArrays_Request & msg)
  : msg_(msg)
  {}
  ::orbbec_camera_msgs::srv::SetArrays_Request data_param(::orbbec_camera_msgs::srv::SetArrays_Request::_data_param_type arg)
  {
    msg_.data_param = std::move(arg);
    return std::move(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Request msg_;
};

class Init_SetArrays_Request_enable
{
public:
  Init_SetArrays_Request_enable()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetArrays_Request_data_param enable(::orbbec_camera_msgs::srv::SetArrays_Request::_enable_type arg)
  {
    msg_.enable = std::move(arg);
    return Init_SetArrays_Request_data_param(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::orbbec_camera_msgs::srv::SetArrays_Request>()
{
  return orbbec_camera_msgs::srv::builder::Init_SetArrays_Request_enable();
}

}  // namespace orbbec_camera_msgs


namespace orbbec_camera_msgs
{

namespace srv
{

namespace builder
{

class Init_SetArrays_Response_message
{
public:
  explicit Init_SetArrays_Response_message(::orbbec_camera_msgs::srv::SetArrays_Response & msg)
  : msg_(msg)
  {}
  ::orbbec_camera_msgs::srv::SetArrays_Response message(::orbbec_camera_msgs::srv::SetArrays_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Response msg_;
};

class Init_SetArrays_Response_success
{
public:
  Init_SetArrays_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetArrays_Response_message success(::orbbec_camera_msgs::srv::SetArrays_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetArrays_Response_message(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::orbbec_camera_msgs::srv::SetArrays_Response>()
{
  return orbbec_camera_msgs::srv::builder::Init_SetArrays_Response_success();
}

}  // namespace orbbec_camera_msgs


namespace orbbec_camera_msgs
{

namespace srv
{

namespace builder
{

class Init_SetArrays_Event_response
{
public:
  explicit Init_SetArrays_Event_response(::orbbec_camera_msgs::srv::SetArrays_Event & msg)
  : msg_(msg)
  {}
  ::orbbec_camera_msgs::srv::SetArrays_Event response(::orbbec_camera_msgs::srv::SetArrays_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Event msg_;
};

class Init_SetArrays_Event_request
{
public:
  explicit Init_SetArrays_Event_request(::orbbec_camera_msgs::srv::SetArrays_Event & msg)
  : msg_(msg)
  {}
  Init_SetArrays_Event_response request(::orbbec_camera_msgs::srv::SetArrays_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_SetArrays_Event_response(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Event msg_;
};

class Init_SetArrays_Event_info
{
public:
  Init_SetArrays_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetArrays_Event_request info(::orbbec_camera_msgs::srv::SetArrays_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_SetArrays_Event_request(msg_);
  }

private:
  ::orbbec_camera_msgs::srv::SetArrays_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::orbbec_camera_msgs::srv::SetArrays_Event>()
{
  return orbbec_camera_msgs::srv::builder::Init_SetArrays_Event_info();
}

}  // namespace orbbec_camera_msgs

#endif  // ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_ARRAYS__BUILDER_HPP_
