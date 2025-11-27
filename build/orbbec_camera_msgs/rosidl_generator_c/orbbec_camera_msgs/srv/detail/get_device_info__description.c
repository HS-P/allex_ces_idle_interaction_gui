// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from orbbec_camera_msgs:srv/GetDeviceInfo.idl
// generated code does not contain a copyright notice

#include "orbbec_camera_msgs/srv/detail/get_device_info__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__GetDeviceInfo__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xc7, 0x6a, 0xb5, 0xff, 0xb3, 0xbb, 0xcf, 0x3f,
      0xc5, 0x46, 0x69, 0xef, 0x0a, 0x52, 0xef, 0x13,
      0x60, 0x71, 0xf2, 0xce, 0xc7, 0x23, 0x13, 0x55,
      0x98, 0xf3, 0x3b, 0x07, 0x67, 0x3f, 0xd8, 0xe0,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe7, 0xf0, 0x1a, 0xfb, 0x64, 0x3d, 0x20, 0x46,
      0x23, 0x0c, 0xb1, 0x36, 0xc2, 0xf8, 0xdc, 0xdc,
      0x43, 0xd3, 0xe6, 0x13, 0x2f, 0xb6, 0x48, 0x45,
      0x1e, 0x49, 0xa9, 0x8f, 0x07, 0x19, 0x86, 0x76,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4a, 0x72, 0xe7, 0x53, 0xe8, 0xd3, 0x0d, 0x1c,
      0xa7, 0xbc, 0x1b, 0x65, 0xa9, 0xbc, 0x21, 0xb7,
      0x79, 0x21, 0x75, 0xfb, 0x0a, 0x30, 0x14, 0xeb,
      0xf7, 0x82, 0x56, 0xfe, 0x79, 0x0f, 0x80, 0xa9,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x5a, 0x95, 0x40, 0x09, 0x57, 0x73, 0xd2, 0x71,
      0x3c, 0xd7, 0x51, 0x2e, 0x93, 0x73, 0xe3, 0xc1,
      0x70, 0xa3, 0xa9, 0xf1, 0x4b, 0xd5, 0xc0, 0xd4,
      0xac, 0xf0, 0x97, 0xf0, 0x12, 0xd5, 0xd1, 0x87,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "orbbec_camera_msgs/msg/detail/device_info__functions.h"
#include "std_msgs/msg/detail/header__functions.h"
#include "service_msgs/msg/detail/service_event_info__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t orbbec_camera_msgs__msg__DeviceInfo__EXPECTED_HASH = {1, {
    0x54, 0x74, 0xd1, 0xa6, 0x68, 0xca, 0x31, 0x50,
    0xcc, 0x01, 0x31, 0xc5, 0xfb, 0x07, 0x25, 0x62,
    0x0a, 0x9a, 0xe0, 0xa2, 0x34, 0x99, 0x1c, 0x31,
    0xc1, 0x91, 0xd7, 0xc5, 0xd1, 0x71, 0xd9, 0x59,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
static const rosidl_type_hash_t std_msgs__msg__Header__EXPECTED_HASH = {1, {
    0xf4, 0x9f, 0xb3, 0xae, 0x2c, 0xf0, 0x70, 0xf7,
    0x93, 0x64, 0x5f, 0xf7, 0x49, 0x68, 0x3a, 0xc6,
    0xb0, 0x62, 0x03, 0xe4, 0x1c, 0x89, 0x1e, 0x17,
    0x70, 0x1b, 0x1c, 0xb5, 0x97, 0xce, 0x6a, 0x01,
  }};
#endif

static char orbbec_camera_msgs__srv__GetDeviceInfo__TYPE_NAME[] = "orbbec_camera_msgs/srv/GetDeviceInfo";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char orbbec_camera_msgs__msg__DeviceInfo__TYPE_NAME[] = "orbbec_camera_msgs/msg/DeviceInfo";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Event__TYPE_NAME[] = "orbbec_camera_msgs/srv/GetDeviceInfo_Event";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME[] = "orbbec_camera_msgs/srv/GetDeviceInfo_Request";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME[] = "orbbec_camera_msgs/srv/GetDeviceInfo_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__GetDeviceInfo__FIELD_NAME__request_message[] = "request_message";
static char orbbec_camera_msgs__srv__GetDeviceInfo__FIELD_NAME__response_message[] = "response_message";
static char orbbec_camera_msgs__srv__GetDeviceInfo__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__GetDeviceInfo__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME, 44, 44},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME, 45, 45},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__GetDeviceInfo_Event__TYPE_NAME, 42, 42},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__GetDeviceInfo__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__DeviceInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Event__TYPE_NAME, 42, 42},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME, 44, 44},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME, 45, 45},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__GetDeviceInfo__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__GetDeviceInfo__TYPE_NAME, 36, 36},
      {orbbec_camera_msgs__srv__GetDeviceInfo__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__GetDeviceInfo__REFERENCED_TYPE_DESCRIPTIONS, 7, 7},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&orbbec_camera_msgs__msg__DeviceInfo__EXPECTED_HASH, orbbec_camera_msgs__msg__DeviceInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__msg__DeviceInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[5].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[6].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__GetDeviceInfo_Request__FIELD_NAME__structure_needs_at_least_one_member[] = "structure_needs_at_least_one_member";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__GetDeviceInfo_Request__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Request__FIELD_NAME__structure_needs_at_least_one_member, 35, 35},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME, 44, 44},
      {orbbec_camera_msgs__srv__GetDeviceInfo_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELD_NAME__info[] = "info";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELD_NAME__success[] = "success";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__msg__DeviceInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELD_NAME__message, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__GetDeviceInfo_Response__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__DeviceInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME, 45, 45},
      {orbbec_camera_msgs__srv__GetDeviceInfo_Response__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__REFERENCED_TYPE_DESCRIPTIONS, 3, 3},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&orbbec_camera_msgs__msg__DeviceInfo__EXPECTED_HASH, orbbec_camera_msgs__msg__DeviceInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__msg__DeviceInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[2].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELD_NAME__info[] = "info";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELD_NAME__request[] = "request";
static char orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME, 44, 44},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME, 45, 45},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__GetDeviceInfo_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__DeviceInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME, 44, 44},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME, 45, 45},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__GetDeviceInfo_Event__TYPE_NAME, 42, 42},
      {orbbec_camera_msgs__srv__GetDeviceInfo_Event__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__GetDeviceInfo_Event__REFERENCED_TYPE_DESCRIPTIONS, 6, 6},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&orbbec_camera_msgs__msg__DeviceInfo__EXPECTED_HASH, orbbec_camera_msgs__msg__DeviceInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__msg__DeviceInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[5].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "---\n"
  "DeviceInfo info\n"
  "bool success\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__GetDeviceInfo__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__GetDeviceInfo__TYPE_NAME, 36, 36},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 47, 47},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Request__TYPE_NAME, 44, 44},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Response__TYPE_NAME, 45, 45},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__GetDeviceInfo_Event__TYPE_NAME, 42, 42},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__GetDeviceInfo__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[8];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 8, 8};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__GetDeviceInfo__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__msg__DeviceInfo__get_individual_type_description_source(NULL);
    sources[3] = *orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_individual_type_description_source(NULL);
    sources[4] = *orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_individual_type_description_source(NULL);
    sources[5] = *orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_individual_type_description_source(NULL);
    sources[6] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[7] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[4];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 4, 4};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__msg__DeviceInfo__get_individual_type_description_source(NULL);
    sources[3] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[7];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 7, 7};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__GetDeviceInfo_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__msg__DeviceInfo__get_individual_type_description_source(NULL);
    sources[3] = *orbbec_camera_msgs__srv__GetDeviceInfo_Request__get_individual_type_description_source(NULL);
    sources[4] = *orbbec_camera_msgs__srv__GetDeviceInfo_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[6] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
