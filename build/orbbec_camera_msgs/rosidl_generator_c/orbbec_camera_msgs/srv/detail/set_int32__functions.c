// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from orbbec_camera_msgs:srv/SetInt32.idl
// generated code does not contain a copyright notice
#include "orbbec_camera_msgs/srv/detail/set_int32__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
orbbec_camera_msgs__srv__SetInt32_Request__init(orbbec_camera_msgs__srv__SetInt32_Request * msg)
{
  if (!msg) {
    return false;
  }
  // data
  return true;
}

void
orbbec_camera_msgs__srv__SetInt32_Request__fini(orbbec_camera_msgs__srv__SetInt32_Request * msg)
{
  if (!msg) {
    return;
  }
  // data
}

bool
orbbec_camera_msgs__srv__SetInt32_Request__are_equal(const orbbec_camera_msgs__srv__SetInt32_Request * lhs, const orbbec_camera_msgs__srv__SetInt32_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // data
  if (lhs->data != rhs->data) {
    return false;
  }
  return true;
}

bool
orbbec_camera_msgs__srv__SetInt32_Request__copy(
  const orbbec_camera_msgs__srv__SetInt32_Request * input,
  orbbec_camera_msgs__srv__SetInt32_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // data
  output->data = input->data;
  return true;
}

orbbec_camera_msgs__srv__SetInt32_Request *
orbbec_camera_msgs__srv__SetInt32_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Request * msg = (orbbec_camera_msgs__srv__SetInt32_Request *)allocator.allocate(sizeof(orbbec_camera_msgs__srv__SetInt32_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(orbbec_camera_msgs__srv__SetInt32_Request));
  bool success = orbbec_camera_msgs__srv__SetInt32_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
orbbec_camera_msgs__srv__SetInt32_Request__destroy(orbbec_camera_msgs__srv__SetInt32_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    orbbec_camera_msgs__srv__SetInt32_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
orbbec_camera_msgs__srv__SetInt32_Request__Sequence__init(orbbec_camera_msgs__srv__SetInt32_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Request * data = NULL;

  if (size) {
    data = (orbbec_camera_msgs__srv__SetInt32_Request *)allocator.zero_allocate(size, sizeof(orbbec_camera_msgs__srv__SetInt32_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = orbbec_camera_msgs__srv__SetInt32_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        orbbec_camera_msgs__srv__SetInt32_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
orbbec_camera_msgs__srv__SetInt32_Request__Sequence__fini(orbbec_camera_msgs__srv__SetInt32_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      orbbec_camera_msgs__srv__SetInt32_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

orbbec_camera_msgs__srv__SetInt32_Request__Sequence *
orbbec_camera_msgs__srv__SetInt32_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Request__Sequence * array = (orbbec_camera_msgs__srv__SetInt32_Request__Sequence *)allocator.allocate(sizeof(orbbec_camera_msgs__srv__SetInt32_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = orbbec_camera_msgs__srv__SetInt32_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
orbbec_camera_msgs__srv__SetInt32_Request__Sequence__destroy(orbbec_camera_msgs__srv__SetInt32_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    orbbec_camera_msgs__srv__SetInt32_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
orbbec_camera_msgs__srv__SetInt32_Request__Sequence__are_equal(const orbbec_camera_msgs__srv__SetInt32_Request__Sequence * lhs, const orbbec_camera_msgs__srv__SetInt32_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!orbbec_camera_msgs__srv__SetInt32_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
orbbec_camera_msgs__srv__SetInt32_Request__Sequence__copy(
  const orbbec_camera_msgs__srv__SetInt32_Request__Sequence * input,
  orbbec_camera_msgs__srv__SetInt32_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(orbbec_camera_msgs__srv__SetInt32_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    orbbec_camera_msgs__srv__SetInt32_Request * data =
      (orbbec_camera_msgs__srv__SetInt32_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!orbbec_camera_msgs__srv__SetInt32_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          orbbec_camera_msgs__srv__SetInt32_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!orbbec_camera_msgs__srv__SetInt32_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
orbbec_camera_msgs__srv__SetInt32_Response__init(orbbec_camera_msgs__srv__SetInt32_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    orbbec_camera_msgs__srv__SetInt32_Response__fini(msg);
    return false;
  }
  return true;
}

void
orbbec_camera_msgs__srv__SetInt32_Response__fini(orbbec_camera_msgs__srv__SetInt32_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
orbbec_camera_msgs__srv__SetInt32_Response__are_equal(const orbbec_camera_msgs__srv__SetInt32_Response * lhs, const orbbec_camera_msgs__srv__SetInt32_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
orbbec_camera_msgs__srv__SetInt32_Response__copy(
  const orbbec_camera_msgs__srv__SetInt32_Response * input,
  orbbec_camera_msgs__srv__SetInt32_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

orbbec_camera_msgs__srv__SetInt32_Response *
orbbec_camera_msgs__srv__SetInt32_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Response * msg = (orbbec_camera_msgs__srv__SetInt32_Response *)allocator.allocate(sizeof(orbbec_camera_msgs__srv__SetInt32_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(orbbec_camera_msgs__srv__SetInt32_Response));
  bool success = orbbec_camera_msgs__srv__SetInt32_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
orbbec_camera_msgs__srv__SetInt32_Response__destroy(orbbec_camera_msgs__srv__SetInt32_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    orbbec_camera_msgs__srv__SetInt32_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
orbbec_camera_msgs__srv__SetInt32_Response__Sequence__init(orbbec_camera_msgs__srv__SetInt32_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Response * data = NULL;

  if (size) {
    data = (orbbec_camera_msgs__srv__SetInt32_Response *)allocator.zero_allocate(size, sizeof(orbbec_camera_msgs__srv__SetInt32_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = orbbec_camera_msgs__srv__SetInt32_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        orbbec_camera_msgs__srv__SetInt32_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
orbbec_camera_msgs__srv__SetInt32_Response__Sequence__fini(orbbec_camera_msgs__srv__SetInt32_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      orbbec_camera_msgs__srv__SetInt32_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

orbbec_camera_msgs__srv__SetInt32_Response__Sequence *
orbbec_camera_msgs__srv__SetInt32_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Response__Sequence * array = (orbbec_camera_msgs__srv__SetInt32_Response__Sequence *)allocator.allocate(sizeof(orbbec_camera_msgs__srv__SetInt32_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = orbbec_camera_msgs__srv__SetInt32_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
orbbec_camera_msgs__srv__SetInt32_Response__Sequence__destroy(orbbec_camera_msgs__srv__SetInt32_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    orbbec_camera_msgs__srv__SetInt32_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
orbbec_camera_msgs__srv__SetInt32_Response__Sequence__are_equal(const orbbec_camera_msgs__srv__SetInt32_Response__Sequence * lhs, const orbbec_camera_msgs__srv__SetInt32_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!orbbec_camera_msgs__srv__SetInt32_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
orbbec_camera_msgs__srv__SetInt32_Response__Sequence__copy(
  const orbbec_camera_msgs__srv__SetInt32_Response__Sequence * input,
  orbbec_camera_msgs__srv__SetInt32_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(orbbec_camera_msgs__srv__SetInt32_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    orbbec_camera_msgs__srv__SetInt32_Response * data =
      (orbbec_camera_msgs__srv__SetInt32_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!orbbec_camera_msgs__srv__SetInt32_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          orbbec_camera_msgs__srv__SetInt32_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!orbbec_camera_msgs__srv__SetInt32_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "orbbec_camera_msgs/srv/detail/set_int32__functions.h"

bool
orbbec_camera_msgs__srv__SetInt32_Event__init(orbbec_camera_msgs__srv__SetInt32_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    orbbec_camera_msgs__srv__SetInt32_Event__fini(msg);
    return false;
  }
  // request
  if (!orbbec_camera_msgs__srv__SetInt32_Request__Sequence__init(&msg->request, 0)) {
    orbbec_camera_msgs__srv__SetInt32_Event__fini(msg);
    return false;
  }
  // response
  if (!orbbec_camera_msgs__srv__SetInt32_Response__Sequence__init(&msg->response, 0)) {
    orbbec_camera_msgs__srv__SetInt32_Event__fini(msg);
    return false;
  }
  return true;
}

void
orbbec_camera_msgs__srv__SetInt32_Event__fini(orbbec_camera_msgs__srv__SetInt32_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  orbbec_camera_msgs__srv__SetInt32_Request__Sequence__fini(&msg->request);
  // response
  orbbec_camera_msgs__srv__SetInt32_Response__Sequence__fini(&msg->response);
}

bool
orbbec_camera_msgs__srv__SetInt32_Event__are_equal(const orbbec_camera_msgs__srv__SetInt32_Event * lhs, const orbbec_camera_msgs__srv__SetInt32_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!orbbec_camera_msgs__srv__SetInt32_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!orbbec_camera_msgs__srv__SetInt32_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
orbbec_camera_msgs__srv__SetInt32_Event__copy(
  const orbbec_camera_msgs__srv__SetInt32_Event * input,
  orbbec_camera_msgs__srv__SetInt32_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!orbbec_camera_msgs__srv__SetInt32_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!orbbec_camera_msgs__srv__SetInt32_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

orbbec_camera_msgs__srv__SetInt32_Event *
orbbec_camera_msgs__srv__SetInt32_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Event * msg = (orbbec_camera_msgs__srv__SetInt32_Event *)allocator.allocate(sizeof(orbbec_camera_msgs__srv__SetInt32_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(orbbec_camera_msgs__srv__SetInt32_Event));
  bool success = orbbec_camera_msgs__srv__SetInt32_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
orbbec_camera_msgs__srv__SetInt32_Event__destroy(orbbec_camera_msgs__srv__SetInt32_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    orbbec_camera_msgs__srv__SetInt32_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
orbbec_camera_msgs__srv__SetInt32_Event__Sequence__init(orbbec_camera_msgs__srv__SetInt32_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Event * data = NULL;

  if (size) {
    data = (orbbec_camera_msgs__srv__SetInt32_Event *)allocator.zero_allocate(size, sizeof(orbbec_camera_msgs__srv__SetInt32_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = orbbec_camera_msgs__srv__SetInt32_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        orbbec_camera_msgs__srv__SetInt32_Event__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
orbbec_camera_msgs__srv__SetInt32_Event__Sequence__fini(orbbec_camera_msgs__srv__SetInt32_Event__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      orbbec_camera_msgs__srv__SetInt32_Event__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

orbbec_camera_msgs__srv__SetInt32_Event__Sequence *
orbbec_camera_msgs__srv__SetInt32_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  orbbec_camera_msgs__srv__SetInt32_Event__Sequence * array = (orbbec_camera_msgs__srv__SetInt32_Event__Sequence *)allocator.allocate(sizeof(orbbec_camera_msgs__srv__SetInt32_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = orbbec_camera_msgs__srv__SetInt32_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
orbbec_camera_msgs__srv__SetInt32_Event__Sequence__destroy(orbbec_camera_msgs__srv__SetInt32_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    orbbec_camera_msgs__srv__SetInt32_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
orbbec_camera_msgs__srv__SetInt32_Event__Sequence__are_equal(const orbbec_camera_msgs__srv__SetInt32_Event__Sequence * lhs, const orbbec_camera_msgs__srv__SetInt32_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!orbbec_camera_msgs__srv__SetInt32_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
orbbec_camera_msgs__srv__SetInt32_Event__Sequence__copy(
  const orbbec_camera_msgs__srv__SetInt32_Event__Sequence * input,
  orbbec_camera_msgs__srv__SetInt32_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(orbbec_camera_msgs__srv__SetInt32_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    orbbec_camera_msgs__srv__SetInt32_Event * data =
      (orbbec_camera_msgs__srv__SetInt32_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!orbbec_camera_msgs__srv__SetInt32_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          orbbec_camera_msgs__srv__SetInt32_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!orbbec_camera_msgs__srv__SetInt32_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
