// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from orbbec_camera_msgs:srv/SetFilter.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "orbbec_camera_msgs/srv/set_filter.hpp"


#ifndef ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_FILTER__TRAITS_HPP_
#define ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_FILTER__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "orbbec_camera_msgs/srv/detail/set_filter__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace orbbec_camera_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetFilter_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: filter_name
  {
    out << "filter_name: ";
    rosidl_generator_traits::value_to_yaml(msg.filter_name, out);
    out << ", ";
  }

  // member: filter_enable
  {
    out << "filter_enable: ";
    rosidl_generator_traits::value_to_yaml(msg.filter_enable, out);
    out << ", ";
  }

  // member: filter_param
  {
    if (msg.filter_param.size() == 0) {
      out << "filter_param: []";
    } else {
      out << "filter_param: [";
      size_t pending_items = msg.filter_param.size();
      for (auto item : msg.filter_param) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetFilter_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: filter_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "filter_name: ";
    rosidl_generator_traits::value_to_yaml(msg.filter_name, out);
    out << "\n";
  }

  // member: filter_enable
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "filter_enable: ";
    rosidl_generator_traits::value_to_yaml(msg.filter_enable, out);
    out << "\n";
  }

  // member: filter_param
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.filter_param.size() == 0) {
      out << "filter_param: []\n";
    } else {
      out << "filter_param:\n";
      for (auto item : msg.filter_param) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetFilter_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace orbbec_camera_msgs

namespace rosidl_generator_traits
{

[[deprecated("use orbbec_camera_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const orbbec_camera_msgs::srv::SetFilter_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  orbbec_camera_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use orbbec_camera_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const orbbec_camera_msgs::srv::SetFilter_Request & msg)
{
  return orbbec_camera_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<orbbec_camera_msgs::srv::SetFilter_Request>()
{
  return "orbbec_camera_msgs::srv::SetFilter_Request";
}

template<>
inline const char * name<orbbec_camera_msgs::srv::SetFilter_Request>()
{
  return "orbbec_camera_msgs/srv/SetFilter_Request";
}

template<>
struct has_fixed_size<orbbec_camera_msgs::srv::SetFilter_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<orbbec_camera_msgs::srv::SetFilter_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace orbbec_camera_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetFilter_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetFilter_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetFilter_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace orbbec_camera_msgs

namespace rosidl_generator_traits
{

[[deprecated("use orbbec_camera_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const orbbec_camera_msgs::srv::SetFilter_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  orbbec_camera_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use orbbec_camera_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const orbbec_camera_msgs::srv::SetFilter_Response & msg)
{
  return orbbec_camera_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<orbbec_camera_msgs::srv::SetFilter_Response>()
{
  return "orbbec_camera_msgs::srv::SetFilter_Response";
}

template<>
inline const char * name<orbbec_camera_msgs::srv::SetFilter_Response>()
{
  return "orbbec_camera_msgs/srv/SetFilter_Response";
}

template<>
struct has_fixed_size<orbbec_camera_msgs::srv::SetFilter_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<orbbec_camera_msgs::srv::SetFilter_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace orbbec_camera_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetFilter_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetFilter_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetFilter_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace orbbec_camera_msgs

namespace rosidl_generator_traits
{

[[deprecated("use orbbec_camera_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const orbbec_camera_msgs::srv::SetFilter_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  orbbec_camera_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use orbbec_camera_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const orbbec_camera_msgs::srv::SetFilter_Event & msg)
{
  return orbbec_camera_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<orbbec_camera_msgs::srv::SetFilter_Event>()
{
  return "orbbec_camera_msgs::srv::SetFilter_Event";
}

template<>
inline const char * name<orbbec_camera_msgs::srv::SetFilter_Event>()
{
  return "orbbec_camera_msgs/srv/SetFilter_Event";
}

template<>
struct has_fixed_size<orbbec_camera_msgs::srv::SetFilter_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Event>
  : std::integral_constant<bool, has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Request>::value && has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<orbbec_camera_msgs::srv::SetFilter_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<orbbec_camera_msgs::srv::SetFilter>()
{
  return "orbbec_camera_msgs::srv::SetFilter";
}

template<>
inline const char * name<orbbec_camera_msgs::srv::SetFilter>()
{
  return "orbbec_camera_msgs/srv/SetFilter";
}

template<>
struct has_fixed_size<orbbec_camera_msgs::srv::SetFilter>
  : std::integral_constant<
    bool,
    has_fixed_size<orbbec_camera_msgs::srv::SetFilter_Request>::value &&
    has_fixed_size<orbbec_camera_msgs::srv::SetFilter_Response>::value
  >
{
};

template<>
struct has_bounded_size<orbbec_camera_msgs::srv::SetFilter>
  : std::integral_constant<
    bool,
    has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Request>::value &&
    has_bounded_size<orbbec_camera_msgs::srv::SetFilter_Response>::value
  >
{
};

template<>
struct is_service<orbbec_camera_msgs::srv::SetFilter>
  : std::true_type
{
};

template<>
struct is_service_request<orbbec_camera_msgs::srv::SetFilter_Request>
  : std::true_type
{
};

template<>
struct is_service_response<orbbec_camera_msgs::srv::SetFilter_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // ORBBEC_CAMERA_MSGS__SRV__DETAIL__SET_FILTER__TRAITS_HPP_
