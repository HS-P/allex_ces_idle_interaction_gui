// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from orbbec_camera_msgs:srv/SetArrays.idl
// generated code does not contain a copyright notice

#include "orbbec_camera_msgs/srv/detail/set_arrays__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetArrays__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x69, 0x34, 0xc9, 0x27, 0x10, 0x67, 0x1f, 0xae,
      0xdb, 0x8d, 0x13, 0xce, 0x3e, 0xe4, 0x26, 0xd5,
      0x92, 0x0d, 0x28, 0xac, 0x85, 0xf4, 0x8d, 0x8a,
      0x3f, 0x79, 0x92, 0xb1, 0xef, 0xbf, 0x15, 0x9a,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetArrays_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x97, 0x7f, 0xda, 0xe8, 0x6f, 0x17, 0x6d, 0x87,
      0x9f, 0x4e, 0xf3, 0xa8, 0xae, 0x83, 0xfa, 0x7c,
      0x27, 0x87, 0x05, 0xf0, 0x0d, 0xa3, 0xcb, 0x8e,
      0xa4, 0x8f, 0x1e, 0x5f, 0xe1, 0xe6, 0x96, 0x6f,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetArrays_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x49, 0x90, 0x6b, 0x63, 0x72, 0x0e, 0xdc, 0xce,
      0xe0, 0x34, 0x80, 0x0e, 0x4e, 0xaf, 0x03, 0x8e,
      0x1b, 0x4a, 0x97, 0x7c, 0x0e, 0x0b, 0x79, 0xd9,
      0x54, 0xb3, 0x60, 0xad, 0xf4, 0x12, 0xec, 0x09,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetArrays_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x8f, 0x27, 0x60, 0x11, 0xef, 0x6f, 0xcc, 0x9f,
      0xb3, 0xba, 0x33, 0xf2, 0x98, 0x2a, 0xde, 0x0b,
      0xd7, 0x4d, 0x19, 0xfd, 0xfc, 0x76, 0xca, 0x80,
      0x5e, 0xaa, 0xbe, 0x8c, 0xe9, 0x70, 0xfc, 0x7d,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char orbbec_camera_msgs__srv__SetArrays__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetArrays";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char orbbec_camera_msgs__srv__SetArrays_Event__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetArrays_Event";
static char orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetArrays_Request";
static char orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetArrays_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetArrays__FIELD_NAME__request_message[] = "request_message";
static char orbbec_camera_msgs__srv__SetArrays__FIELD_NAME__response_message[] = "response_message";
static char orbbec_camera_msgs__srv__SetArrays__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetArrays__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetArrays__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME, 41, 41},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__SetArrays_Event__TYPE_NAME, 38, 38},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__SetArrays__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Event__TYPE_NAME, 38, 38},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME, 41, 41},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetArrays__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetArrays__TYPE_NAME, 32, 32},
      {orbbec_camera_msgs__srv__SetArrays__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__SetArrays__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__srv__SetArrays_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = orbbec_camera_msgs__srv__SetArrays_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = orbbec_camera_msgs__srv__SetArrays_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetArrays_Request__FIELD_NAME__enable[] = "enable";
static char orbbec_camera_msgs__srv__SetArrays_Request__FIELD_NAME__data_param[] = "data_param";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetArrays_Request__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetArrays_Request__FIELD_NAME__enable, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Request__FIELD_NAME__data_param, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetArrays_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME, 40, 40},
      {orbbec_camera_msgs__srv__SetArrays_Request__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetArrays_Response__FIELD_NAME__success[] = "success";
static char orbbec_camera_msgs__srv__SetArrays_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetArrays_Response__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetArrays_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Response__FIELD_NAME__message, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetArrays_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME, 41, 41},
      {orbbec_camera_msgs__srv__SetArrays_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetArrays_Event__FIELD_NAME__info[] = "info";
static char orbbec_camera_msgs__srv__SetArrays_Event__FIELD_NAME__request[] = "request";
static char orbbec_camera_msgs__srv__SetArrays_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetArrays_Event__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetArrays_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME, 41, 41},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__SetArrays_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME, 41, 41},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetArrays_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetArrays_Event__TYPE_NAME, 38, 38},
      {orbbec_camera_msgs__srv__SetArrays_Event__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__SetArrays_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__srv__SetArrays_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = orbbec_camera_msgs__srv__SetArrays_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "bool enable\n"
  "float32[] data_param\n"
  "---\n"
  "bool success\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetArrays__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetArrays__TYPE_NAME, 32, 32},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 64, 64},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetArrays_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetArrays_Request__TYPE_NAME, 40, 40},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetArrays_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetArrays_Response__TYPE_NAME, 41, 41},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetArrays_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetArrays_Event__TYPE_NAME, 38, 38},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetArrays__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetArrays__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__srv__SetArrays_Event__get_individual_type_description_source(NULL);
    sources[3] = *orbbec_camera_msgs__srv__SetArrays_Request__get_individual_type_description_source(NULL);
    sources[4] = *orbbec_camera_msgs__srv__SetArrays_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetArrays_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetArrays_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetArrays_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetArrays_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetArrays_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetArrays_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__srv__SetArrays_Request__get_individual_type_description_source(NULL);
    sources[3] = *orbbec_camera_msgs__srv__SetArrays_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
