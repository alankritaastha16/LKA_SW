#!/usr/bin/python3

#image version
VERSION = "2.0.1"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
from ros_robot_controller_msgs.msg import MotorsState, MotorState
from ultralytics import YOLO
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time 
from std_msgs.msg import Int32, Float64
import yaml


class LineDetectionNode(Node):
    def __init__(self):
        super().__init__('line_detect')

        self.declare_parameter('hsv_config_path', '')
        self.declare_parameter('debug_views', True)
        self.declare_parameter('obstacle_height',0.3)
        self.declare_parameter('go_straight_time_left',1.5)
        self.declare_parameter('base_speed',0.5)
        self.declare_parameter('kp',0.01)
        self.declare_parameter('go_straight_time_right',1.5)
        self.declare_parameter('motor_command_2_time_right',1.0)

        hsv_config_path = self.get_parameter('hsv_config_path').get_parameter_value().string_value
        self.obstacle_height = self.get_parameter('obstacle_height').value
        self.go_straight_time_left = self.get_parameter('go_straight_time_left').value
        self.base_speed = self.get_parameter('base_speed').value
        self.kp = self.get_parameter('kp').value
        self.go_straight_time_right = self.get_parameter('go_straight_time_right').value
        self.motor_command_2_time_right = self.get_parameter('motor_command_2_time_right').value


        if hsv_config_path:
            with open(hsv_config_path, 'r') as f:
                hsv_config = yaml.safe_load(f)
                self.LOWER_WHITE = tuple(hsv_config['hsv']['lower'])
                self.UPPER_WHITE = tuple(hsv_config['hsv']['upper'])
        else:
            self.LOWER_WHITE, self.UPPER_WHITE = [(27, 0, 243),(93, 255, 255)]

        # Load YOLO models
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        self.object_model = YOLO("/home/de-hiwon24-rob2/drive_ws/src/line_detection/config/cones_roboflow.pt")  # object detection

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        self.bridge = CvBridge()
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos)
        self.motor_pub = self.create_publisher(MotorsState, '/ros_robot_controller_node/set_motor', 10)
        self.obstacle_pub = self.create_publisher(Int32, '/obstacle', 10)
        self.engine_rpm_pub = self.create_publisher(Float64, '/engine_rpm', 10)


        self.motor_enabled = False
        self.smoothed_midpoint = None
        self.last_valid_midpoint = None
        self.EMA_ALPHA = 0.2

        self.line_lost_time = None
        self.line_lost_timeout = 0.5  # second

        self.mode = 'LANE_FOLLOWING'  # or 'OVERTAKING'
        self.overtake_path = []
        self.overtake_index = 0

        self.obstacle_detected = Int32()
        self.obstacle_detected.data=0

        self.engine_rpm = Float64()
        self.engine_rpm.data=0.0

        self.get_logger().info('Line Detection Node Started')

    def joy_callback(self, msg):
        l2 = msg.buttons[6]
        r2 = msg.buttons[7]

        if l2:
            if not self.motor_enabled:
                self.get_logger().info("Motors enabled via L2")
            self.motor_enabled = True
            self.line_lost_time = None

        if r2:
            if self.motor_enabled:
                self.get_logger().info("Motors stopped via R2")
            self.motor_enabled = False
            self.overtake_index = 0
            self.stop_motors()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_copy = frame.copy()
        # print(f"motor_command_1_time: {self.motor_command_1_time}, go_straight_time: {self.go_straight_time},motor_command_2_time: {self.motor_command_2_time}")

        # Lane detection and/or overtaking path rendering
        output_frame, midpoint, left_point, right_point = self.process_frame(frame_copy)
        height, width = frame.shape[:2]

        if not self.motor_enabled:
            if output_frame is not None:
                cv2.imshow('Line Detection Debug', output_frame)
                cv2.waitKey(1)
            return

        # Object Detection (Overtake Trigger)
        object_results = self.object_model.predict(frame, conf=0.7, iou=0.3, verbose=False)[0]
        detected_objects = []
        if object_results.boxes is not None:
             for box, cls_id in zip(object_results.boxes.xyxy.cpu().numpy(), object_results.boxes.cls.cpu().numpy()):
                if int(cls_id) == 0:  #0 for cones
                    x1, y1, x2, y2 = box.astype(int)

                    # Draw boundingbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    if y2 > height * self.obstacle_height:
                        detected_objects.append((x1, y1, x2, y2))
                        self.obstacle_detected.data=1
                        self.obstacle_pub.publish(self.obstacle_detected)
                        #time.sleep(2.0)

        if self.mode == 'LANE_FOLLOWING' and detected_objects:
            self.get_logger().info("Obstacle detected, switching to overtaking mode.")
            self.mode = 'OVERTAKING'

        if self.mode == 'OVERTAKING':
            if self.overtake_index == 0:
                self.send_motor_command_smooth(20, width // 2,0)
                # time.sleep(0.001)
                self.go_straight_command()
                time.sleep(self.go_straight_time_left)
                # self.send_motor_command_smooth(20, width // 2,1)
                

            elif self.overtake_index == 1:
                # self.send_motor_command(20, width // 2,1)
                self.send_motor_command_smooth(20, width // 2,1)
                self.go_straight_command()
                time.sleep(self.go_straight_time_right)

            self.get_logger().info("-------------------")
            self.get_logger().info("Overtake complete. Resuming lane following.")
            self.mode = 'LANE_FOLLOWING'
            self.obstacle_detected.data = 0
            if self.overtake_index == 1:
                self.overtake_index = 0
            else:
                self.overtake_index = 1
            return

        # *****Lane Following******
        output_frame, midpoint, left_point, right_point = self.process_frame(frame)
        current_time = time.time()

        if midpoint and (left_point or right_point):
            self.line_lost_time = None
            self.send_motor_command(midpoint[0], width // 2,0)
        else:
            if self.line_lost_time is None:
                self.line_lost_time = current_time
            elif current_time - self.line_lost_time > self.line_lost_timeout:
                self.get_logger().info("Line lost. Stopping motors.")
                self.stop_motors()
        if output_frame is not None:
            cv2.imshow('Line Detection Debug', output_frame)
            cv2.waitKey(1)

    def detect_edges(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def region_of_interest(self, img):
        height, width = img.shape[:2]
        mask = np.zeros_like(img)
        polygon = np.array([[
            (0, int(height * 0.5)),
            (width, int(height * 0.5)),
            (width, int(height * 0.4)),
            (0, int(height * 0.4))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def get_stable_top_point(self, points, side="left", center_x=640, cluster_height=5):
        side_points = points[points[:, 0] < center_x] if side == "left" else points[points[:, 0] > center_x]
        if len(side_points) == 0:
            return None
        min_y = np.min(side_points[:, 1])
        # cluster = side_points[(side_points[:, 1] >= min_y) & (side_points[:, 1] <= min_y + cluster_height)]
        # avg_x = int(np.mean(cluster[:, 0]))
        # avg_y = int(np.mean(cluster[:, 1]))
        # return (avg_x, avg_y)
        # return (avg_x, avg_y)
        # return (avg_x, avg_)
        # Keep only top-most points
        top_points = side_points[side_points[:, 1] == min_y]
        
        # Find the point closest to center_x
        if side == "left":
            best_point = top_points[np.argmax(top_points[:, 0])]  # largest x (closest to center)
        else:
            best_point = top_points[np.argmin(top_points[:, 0])]  # smallest x (closest to center)
        
        return tuple(best_point)

    def stack_frames(self, frames, width=800, height=600):
        resized = [cv2.resize(f, (width, height)) for f in frames if f is not None and f.shape[0] > 0 and f.shape[1] > 0]
        return np.hstack(resized) if resized else None

    def resize_output(self, image, max_width=1280):
        if image is None:
            return None
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.LOWER_WHITE, self.UPPER_WHITE)
        white_masked = cv2.bitwise_and(frame, frame, mask=white_mask)

        gray = cv2.cvtColor(white_masked, cv2.COLOR_BGR2GRAY)
        edges = self.detect_edges(gray)
        roi = self.region_of_interest(edges)

        height, width = frame.shape[:2]

        frame2=cv2.rectangle(frame,(0, int(height*0.5 )),(width, int(height * 0.4)), (255,0,0),4)

        center_x = width // 2
        points = np.flip(np.column_stack(np.where(roi > 0)), axis=1)
        left_point = self.get_stable_top_point(points, "left", center_x)
        right_point = self.get_stable_top_point(points, "right", center_x)
        vis = frame.copy()

        if left_point and right_point:
            raw_mid = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
        elif left_point:
            raw_mid = (left_point[0] + 300, left_point[1])
        elif right_point:
            raw_mid = (right_point[0] - 300, right_point[1])
        else:
            raw_mid = self.last_valid_midpoint if self.last_valid_midpoint else (center_x, int(height * 0.6))

        if self.smoothed_midpoint is None:
            self.smoothed_midpoint = raw_mid
        else:
            self.smoothed_midpoint = (
                int(self.EMA_ALPHA * raw_mid[0] + (1 - self.EMA_ALPHA) * self.smoothed_midpoint[0]),
                int(self.EMA_ALPHA * raw_mid[1] + (1 - self.EMA_ALPHA) * self.smoothed_midpoint[1])
            )

        if left_point or right_point:
            self.last_valid_midpoint = self.smoothed_midpoint

        cv2.circle(vis, self.smoothed_midpoint, 10, (0, 255, 255), 1)
        if left_point:
            cv2.circle(vis, left_point, 10, (255, 0, 0), -1)
        if right_point:
            cv2.circle(vis, right_point, 10, (0, 0, 255), -1)

        # overlay = vis.copy()
        # width_half = 230
        # driveway_top = self.smoothed_midpoint[1]
        # driveway_bottom = height
        # polygon = np.array([
        #     (self.smoothed_midpoint[0] - width_half, driveway_top),
        #     (self.smoothed_midpoint[0] + width_half, driveway_top),
        #     (self.smoothed_midpoint[0] + width_half, driveway_bottom),
        #     (self.smoothed_midpoint[0] - width_half, driveway_bottom)
        # ], np.int32)
        # cv2.fillPoly(overlay, [polygon]

        white_mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = self.stack_frames([vis, edges_colored, white_mask_bgr])
        return self.resize_output(combined), self.smoothed_midpoint, left_point, right_point

    def send_motor_command_smooth(self, target_x, image_center_x, sign_flag):
        if sign_flag==0:
            error = target_x - image_center_x
            for i in np.arange(0,error,-1.5):
                kp = self.kp
                base_speed = self.base_speed
                steering = kp * (i * 0.25)
                # self.get_logger().info(f'Steering: {steering:.2f}')

                left_speed = base_speed + steering
                right_speed = base_speed - steering

                self.engine_rpm.data= (left_speed if left_speed>right_speed else right_speed)

                msg = MotorsState()
                msg.data = [
                    MotorState(id=1, rps=left_speed),
                    MotorState(id=2, rps=left_speed),
                    MotorState(id=3, rps=-right_speed),
                    MotorState(id=4, rps=-right_speed)
                ]
                self.motor_pub.publish(msg)
                self.engine_rpm_pub.publish(self.engine_rpm)
                time.sleep(2.5/abs(error))

        if sign_flag==1:
            error = -(target_x - image_center_x)
            for i in np.arange(0,error,1.5):
                kp = self.kp
                base_speed = self.base_speed
                steering = kp * (i * 0.25)
                # self.get_logger().info(f'Steering: {steering:.2f}')

                left_speed = base_speed + steering
                right_speed = base_speed - steering

                self.engine_rpm.data= (left_speed if left_speed>right_speed else right_speed)

                msg = MotorsState()
                msg.data = [
                    MotorState(id=1, rps=left_speed),
                    MotorState(id=2, rps=left_speed),
                    MotorState(id=3, rps=-right_speed),
                    MotorState(id=4, rps=-right_speed)
                ]
                self.motor_pub.publish(msg)
                self.engine_rpm_pub.publish(self.engine_rpm)
                time.sleep(2.2/abs(error))
        

    def send_motor_command(self, target_x, image_center_x, sign_flag):
        if sign_flag==0:
            error = target_x - image_center_x
        if sign_flag==1:
            error = -(target_x - image_center_x)
        kp = self.kp
        base_speed = self.base_speed
        steering = kp * (error * 0.25)
        # self.get_logger().info(f'Steering: {steering:.2f}')********************{steering:.2f}'){steering:.2f}')
        # self.get_logger().info(f'Steering: {steering:.2f}')//////

        left_speed = base_speed + steering
        right_speed = base_speed - steering

        self.engine_rpm.data= (left_speed if left_speed>right_speed else right_speed)

        self.get_logger().info(f'engine_rpm, left_speed, right_speed: {self.engine_rpm.data}, {left_speed}, {right_speed}')

        msg = MotorsState()
        msg.data = [
            MotorState(id=1, rps=left_speed),
            MotorState(id=2, rps=left_speed),
            MotorState(id=3, rps=-right_speed),
            MotorState(id=4, rps=-right_speed)
        ]
        self.motor_pub.publish(msg)
        self.engine_rpm_pub.publish(self.engine_rpm)
        



    def go_straight_command(self):
        base_speed = self.base_speed

        left_speed = base_speed 
        right_speed = base_speed 

        msg = MotorsState()
        msg.data = [
            MotorState(id=1, rps=left_speed),
            MotorState(id=2, rps=left_speed),
            MotorState(id=3, rps=-right_speed),
            MotorState(id=4, rps=-right_speed)
        ]
        self.motor_pub.publish(msg)

    def stop_motors(self):
        msg = MotorsState()
        msg.data = [MotorState(id=i, rps=0.0) for i in range(1, 5)]
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (Ctrl+C)')
    finally:
        node.stop_motors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()






