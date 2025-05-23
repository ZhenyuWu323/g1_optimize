# Gepetto

## GUI

run gepetto-gui



这个配置的逻辑是：

前3个DOF（通常是位置x,y,z）：权重为0
接下来3个DOF（通常是姿态roll,pitch,yaw）：权重为500.0
剩余的关节位置：权重为0.01
所有关节速度：权重为10