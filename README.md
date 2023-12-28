### Towards C-V2X Enabled Collaborative Autonomous Driving ###

Abstract: Intelligent vehicles, including autonomous vehicles and vehicles equipped with ADAS systems, are single-agent systems that navigate solely on the information collected by themselves. However, despite rapid advancements in hardware and algorithms, many accidents still occur due to the limited sensing coverage from a single-agent perception angle. These tragedies raise a critical question of whether single-agent autonomous driving is safe. Preliminary investigations on this safety issue led us to create a C-V2X-enabled collaborative autonomous driving framework (CCAD) to observe the driving circumstance from multiple perception angles. Our framework uses C-V2X technology to connect infrastructure with vehicles and vehicles with vehicles to transmit safety-critical information and to add safety redundancies. By enabling these communication channels, we connect previously independent single-agent vehicles and existing infrastructure. This paper presents a prototype of our CCAD framework with RSU and OBU as communication devices and an edge-computing device for data processing. We also present a case study of successfully implementing an infrastructure-based collaborative lane-keeping with the CCAD framework. Our case study evaluations demonstrate that the CCAD framework can transmit, in real-time, personalized lane-keeping guidance information when the vehicle cannot find the lanes. The evaluations also indicate that the CCAD framework can drastically improve the safety of single-agent intelligent vehicles and open the doors to many more collaborative autonomous driving applications.

## What is included in this Repo ##

1. Source Code to replicate the experiment.
2. Converted C-V2X messages from C++ to Python/ROS2 (ROS1 should be compatible, but not tested).
3. Paper

## Cite this Paper ##

# Plain Text #
Y. He, B. Wu, Z. Dong, J. Wan and W. Shi, "Towards C-V2X Enabled Collaborative Autonomous Driving," in IEEE Transactions on Vehicular Technology, vol. 72, no. 12, pp. 15450-15462, Dec. 2023, doi: 10.1109/TVT.2023.3299844.

# Bibtex #
@ARTICLE{10216776,
  author={He, Yuankai and Wu, Baofu and Dong, Zheng and Wan, Jian and Shi, Weisong},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Towards C-V2X Enabled Collaborative Autonomous Driving}, 
  year={2023},
  volume={72},
  number={12},
  pages={15450-15462},
  doi={10.1109/TVT.2023.3299844}}
