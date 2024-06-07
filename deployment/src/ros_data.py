import rclpy
from rclpy.node import Node

class ROSData:
    def __init__(self, node: Node, timeout: int = 3, queue_size: int = 1, name: str = ""):
        self.node = node
        self.timeout = timeout
        self.last_time_received = self.node.get_clock().now()
        self.queue_size = queue_size
        self.data = None
        self.name = name
        self.phantom = False

    def get(self):
        return self.data

    def set(self, data):
        current_time = self.node.get_clock().now()
        time_waited = current_time - self.last_time_received
        time_waited_sec = time_waited.nanoseconds / 1e9  # Convert nanoseconds to seconds

        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited_sec > self.timeout:  # Reset queue if timeout
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)

        self.last_time_received = current_time

    def is_valid(self, verbose: bool = False):
        current_time = self.node.get_clock().now()
        time_waited = current_time - self.last_time_received
        time_waited_sec = time_waited.nanoseconds / 1e9  # Convert nanoseconds to seconds

        valid = time_waited_sec < self.timeout
        if self.queue_size > 1:
            valid = valid and len(self.data) == self.queue_size
        else:
            valid = valid and self.data is not None
        if verbose and not valid:
            print(f"Not receiving {self.name} data for {time_waited_sec} seconds (timeout: {self.timeout} seconds)")
        return valid
