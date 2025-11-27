// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from orbbec_camera_msgs:srv/SetInt32.idl
// generated code does not contain a copyright notice

#include "orbbec_camera_msgs/srv/detail/set_int32__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetInt32__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x79, 0xc6, 0x4e, 0xfc, 0xb6, 0x33, 0x91, 0xf9,
      0x10, 0x57, 0xdb, 0x13, 0x3a, 0x13, 0xf5, 0xa6,
      0xf7, 0x51, 0x04, 0x45, 0xfa, 0x5e, 0x02, 0xd0,
      0xa0, 0xe5, 0x4e, 0x8e, 0xaf, 0x04, 0xbd, 0x8f,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetInt32_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x8f, 0xe1, 0xc4, 0x91, 0x1f, 0xc5, 0xdb, 0x5d,
      0xee, 0x45, 0x72, 0x73, 0x27, 0xb4, 0xe0, 0xdf,
      0x88, 0x4b, 0xc9, 0xaa, 0x39, 0x02, 0xc8, 0x7f,
      0x97, 0x5f, 0x65, 0xa0, 0x65, 0x34, 0xdf, 0xbd,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetInt32_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xd1, 0xf0, 0xc5, 0x29, 0xf0, 0xb6, 0x59, 0xe7,
      0xa0, 0x06, 0xe4, 0xd8, 0x57, 0xf9, 0x1d, 0x44,
      0x35, 0x00, 0xe6, 0x85, 0xc3, 0xb6, 0x00, 0xe9,
      0x59, 0x03, 0xeb, 0x42, 0xeb, 0x13, 0xc0, 0xbb,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__srv__SetInt32_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x6d, 0xa1, 0x4f, 0x88, 0x0f, 0x40, 0xeb, 0xbe,
      0x6d, 0xa7, 0x85, 0xac, 0xc1, 0x13, 0xa3, 0xb6,
      0xe8, 0x96, 0xb5, 0x22, 0x10, 0xf4, 0x04, 0xfe,
      0x74, 0x8d, 0xe5, 0x01, 0xea, 0x28, 0xb8, 0xbd,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "service_msgs/msg/detail/service_event_info__functions.h"

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

static char orbbec_camera_msgs__srv__SetInt32__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetInt32";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char orbbec_camera_msgs__srv__SetInt32_Event__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetInt32_Event";
static char orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetInt32_Request";
static char orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME[] = "orbbec_camera_msgs/srv/SetInt32_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetInt32__FIELD_NAME__request_message[] = "request_message";
static char orbbec_camera_msgs__srv__SetInt32__FIELD_NAME__response_message[] = "response_message";
static char orbbec_camera_msgs__srv__SetInt32__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetInt32__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetInt32__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME, 39, 39},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {orbbec_camera_msgs__srv__SetInt32_Event__TYPE_NAME, 37, 37},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__SetInt32__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Event__TYPE_NAME, 37, 37},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME, 39, 39},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetInt32__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetInt32__TYPE_NAME, 31, 31},
      {orbbec_camera_msgs__srv__SetInt32__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__SetInt32__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__srv__SetInt32_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = orbbec_camera_msgs__srv__SetInt32_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = orbbec_camera_msgs__srv__SetInt32_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetInt32_Request__FIELD_NAME__data[] = "data";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetInt32_Request__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetInt32_Request__FIELD_NAME__data, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetInt32_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME, 39, 39},
      {orbbec_camera_msgs__srv__SetInt32_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetInt32_Response__FIELD_NAME__success[] = "success";
static char orbbec_camera_msgs__srv__SetInt32_Response__FIELD_NAME__message[] = "message";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetInt32_Response__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetInt32_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Response__FIELD_NAME__message, 7, 7},
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
orbbec_camera_msgs__srv__SetInt32_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME, 40, 40},
      {orbbec_camera_msgs__srv__SetInt32_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char orbbec_camera_msgs__srv__SetInt32_Event__FIELD_NAME__info[] = "info";
static char orbbec_camera_msgs__srv__SetInt32_Event__FIELD_NAME__request[] = "request";
static char orbbec_camera_msgs__srv__SetInt32_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__srv__SetInt32_Event__FIELDS[] = {
  {
    {orbbec_camera_msgs__srv__SetInt32_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME, 39, 39},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME, 40, 40},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__srv__SetInt32_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME, 39, 39},
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME, 40, 40},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__srv__SetInt32_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__srv__SetInt32_Event__TYPE_NAME, 37, 37},
      {orbbec_camera_msgs__srv__SetInt32_Event__FIELDS, 3, 3},
    },
    {orbbec_camera_msgs__srv__SetInt32_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = orbbec_camera_msgs__srv__SetInt32_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = orbbec_camera_msgs__srv__SetInt32_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "int32 data\n"
  "---\n"
  "bool success\n"
  "string message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetInt32__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetInt32__TYPE_NAME, 31, 31},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 42, 42},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetInt32_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetInt32_Request__TYPE_NAME, 39, 39},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetInt32_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetInt32_Response__TYPE_NAME, 40, 40},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__srv__SetInt32_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__srv__SetInt32_Event__TYPE_NAME, 37, 37},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetInt32__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetInt32__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__srv__SetInt32_Event__get_individual_type_description_source(NULL);
    sources[3] = *orbbec_camera_msgs__srv__SetInt32_Request__get_individual_type_description_source(NULL);
    sources[4] = *orbbec_camera_msgs__srv__SetInt32_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetInt32_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetInt32_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetInt32_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetInt32_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__srv__SetInt32_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__srv__SetInt32_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *orbbec_camera_msgs__srv__SetInt32_Request__get_individual_type_description_source(NULL);
    sources[3] = *orbbec_camera_msgs__srv__SetInt32_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
