#!/usr/bin/env python3
import json
import rospy
from std_msgs.msg import String

def callback(msg):
    try:
        data = json.loads(msg.data)
        seq   = data.get("header", {}).get("seq", "?")
        stamp = float(data.get("header", {}).get("stamp", rospy.get_time()))
        host  = data.get("host", "?")
        tag   = data.get("tag", "")
        note  = data.get("note", "")
        lat_ms = max(0.0, (rospy.get_time() - stamp) * 1000.0)

        rospy.loginfo("heard seq=%s from %s tag=%s  lat=%.1fms  note='%s'",
                      seq, host, tag, lat_ms, note)
    except Exception:
        rospy.loginfo("heard (raw): %s", msg.data)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("chatter", String, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()




