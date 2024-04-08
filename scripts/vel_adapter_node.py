#!/usr/bin/python3
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class CmdVelAdapter:
    def __init__(self, wheel_base=0.235, wheel_radius=0.035) :
         
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        # Joint state subscriber
        rospy.Subscriber("/cmd_vel", Twist, callback=self.cmd_vel_callback)

        # Passive joint position publisher (ROS control command)
        self.vel_pub = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities", Float64MultiArray, queue_size=1)
        
    def cmd_vel_callback(self, msg):
        #Publish passive joint angles
        msgOut = Float64MultiArray()
        
        left_wheel_vel = msg.linear.x + (msg.angular.z * self.wheel_base / 2) 
        right_wheel_vel = msg.linear.x - (msg.angular.z * self.wheel_base / 2)

        msgOut.data = [right_wheel_vel/self.wheel_radius,left_wheel_vel/self.wheel_radius]
        self.vel_pub.publish(msgOut)

if __name__ == '__main__':
    rospy.init_node("velocity_controller")
    rospy.loginfo("Velocity controller node started")
    try:
        robot = CmdVelAdapter()
    except rospy.ROSInterruptException:
        print("An error occurred")
    rospy.spin()