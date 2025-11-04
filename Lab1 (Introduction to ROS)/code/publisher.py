#!/usr/bin/env python3
import os
import socket
import json
import math
import random

import rospy
from std_msgs.msg import String

robot_quot = [
    "Stay curious. Stay calibrated.",
    "Robots do it deterministically. Mostly.",
    "Latency low, curiosity high.",
    "Safety first, then autonomy.",
    "Sensors up, noise down.",
    "East or West, CS 545 is the best."
]


def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)

    # Node & params
    rospy.init_node('talker', anonymous=True)
    rate_hz = rospy.get_param("~rate_hz", 50)    
    tag = rospy.get_param("~tag", "Team Roomba")    
    rate = rospy.Rate(rate_hz)

    host = socket.gethostname()
    seq = 0

    rospy.loginfo("talker started on host=%s, rate=%s Hz, tag=%s", host, rate_hz, tag)

    while not rospy.is_shutdown():
        seq += 1
        t = rospy.get_time()

        payload = {
            "header": {
                "seq": seq,
                "stamp": t
            },
            "node": rospy.get_name(),
            "host": host,
            "tag": tag,
            "note": random.choice(robot_quot)
        }

        msg = json.dumps(payload, separators=(",", ":"))
        rospy.loginfo(msg)          
        pub.publish(msg)           

        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

