# Methodology

## Passenger Volume Estimation
First we try to count the number of people in a platform to find the current passenger volume of a certain station heading towards a certain direction. We take into account
the metro card punches and the surveillance camera feed.
  ### Computer Vision
  We utilize OpenCV and Ultralytics to estimate headcount in crowded environments, such as station platforms. By combining Rapid Pass data with surveillance camera feeds, 
  we determine the number of passengers present at a specific station and heading in a particular direction. For sparse crowds, we employ the pretrained model 
  Faster R-CNN Inception-ResNet-v2 to detect individuals. If the detected headcount exceeds 50, we automatically switch to CSRNet, which is better suited for dense 
  crowd estimation.
  <img width="1478" height="559" alt="diagram-export-7-24-2025-1_56_15-PM" src="https://github.com/user-attachments/assets/662d1d10-f4c8-44bf-b3dd-ad6dc7aa5944" />

  ### 
