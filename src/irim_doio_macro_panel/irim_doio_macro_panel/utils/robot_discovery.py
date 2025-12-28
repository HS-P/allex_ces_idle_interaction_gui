from __future__ import annotations

import re
import subprocess
from typing import List

def get_robots_from_ros2_topic_list(topic_regex: str) -> List[str]:
    """
    Calls `ros2 topic list` and parses robot names from topics like:
      /robot_outbound_data/<name>/...
    Returns unique names preserving order.
    """
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    lines = result.stdout.splitlines()
    pattern = re.compile(topic_regex)
    seen: List[str] = []
    for line in lines:
        m = pattern.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        if name not in seen:
            seen.append(name)
    return seen
