// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from orbbec_camera_msgs:msg/IMUInfo.idl
// generated code does not contain a copyright notice

#include "orbbec_camera_msgs/msg/detail/imu_info__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_orbbec_camera_msgs
const rosidl_type_hash_t *
orbbec_camera_msgs__msg__IMUInfo__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x02, 0xed, 0x2a, 0xdc, 0x4a, 0x07, 0xb3, 0xcd,
      0xaa, 0x44, 0xa2, 0x84, 0x74, 0x64, 0x80, 0x05,
      0x7f, 0x44, 0xba, 0x47, 0xe2, 0x24, 0x53, 0x31,
      0xa6, 0xab, 0x8f, 0xc7, 0xd2, 0x4b, 0x56, 0x08,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "std_msgs/msg/detail/header__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t std_msgs__msg__Header__EXPECTED_HASH = {1, {
    0xf4, 0x9f, 0xb3, 0xae, 0x2c, 0xf0, 0x70, 0xf7,
    0x93, 0x64, 0x5f, 0xf7, 0x49, 0x68, 0x3a, 0xc6,
    0xb0, 0x62, 0x03, 0xe4, 0x1c, 0x89, 0x1e, 0x17,
    0x70, 0x1b, 0x1c, 0xb5, 0x97, 0xce, 0x6a, 0x01,
  }};
#endif

static char orbbec_camera_msgs__msg__IMUInfo__TYPE_NAME[] = "orbbec_camera_msgs/msg/IMUInfo";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char std_msgs__msg__Header__TYPE_NAME[] = "std_msgs/msg/Header";

// Define type names, field names, and default values
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__header[] = "header";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__noise_density[] = "noise_density";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__random_walk[] = "random_walk";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__reference_temperature[] = "reference_temperature";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__bias[] = "bias";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__gravity[] = "gravity";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__scale_misalignment[] = "scale_misalignment";
static char orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__temperature_slope[] = "temperature_slope";

static rosidl_runtime_c__type_description__Field orbbec_camera_msgs__msg__IMUInfo__FIELDS[] = {
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__header, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__noise_density, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__random_walk, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__reference_temperature, 21, 21},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__bias, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      3,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__gravity, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      3,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__scale_misalignment, 18, 18},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      9,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {orbbec_camera_msgs__msg__IMUInfo__FIELD_NAME__temperature_slope, 17, 17},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE_ARRAY,
      9,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription orbbec_camera_msgs__msg__IMUInfo__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {std_msgs__msg__Header__TYPE_NAME, 19, 19},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
orbbec_camera_msgs__msg__IMUInfo__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {orbbec_camera_msgs__msg__IMUInfo__TYPE_NAME, 30, 30},
      {orbbec_camera_msgs__msg__IMUInfo__FIELDS, 8, 8},
    },
    {orbbec_camera_msgs__msg__IMUInfo__REFERENCED_TYPE_DESCRIPTIONS, 2, 2},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&std_msgs__msg__Header__EXPECTED_HASH, std_msgs__msg__Header__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = std_msgs__msg__Header__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "std_msgs/Header header\n"
  "float64 noise_density\n"
  "float64 random_walk\n"
  "float64 reference_temperature\n"
  "float64[3] bias\n"
  "float64[3] gravity\n"
  "float64[9] scale_misalignment\n"
  "float64[9] temperature_slope\n"
  "";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
orbbec_camera_msgs__msg__IMUInfo__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {orbbec_camera_msgs__msg__IMUInfo__TYPE_NAME, 30, 30},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 190, 190},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
orbbec_camera_msgs__msg__IMUInfo__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[3];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 3, 3};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *orbbec_camera_msgs__msg__IMUInfo__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *std_msgs__msg__Header__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
