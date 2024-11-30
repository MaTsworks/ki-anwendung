from robomaster import robot

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="ap", proto_type="udp")

ep_chassis = ep_robot.chassis
ep_chassis.move(x=1, y=0, z=0, xy_speed=1).wait_for_completed()

ep_robot.close()
