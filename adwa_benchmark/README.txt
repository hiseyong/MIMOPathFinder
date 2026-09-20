This is the dataset used for evaluation of the navigation in the paper "Adaptive  Dynamic  Window  Approach  for  Local  Navigation" published at IROS 2020.
You can use these files as you wish.
Matej Dobrevski, matej.dobrevski@fri.uni-lj.si, Visual Cognitive Systems Laboratory, Faculty of Comupter and Information Science, University of Ljubljana.

File description:
1. The .pgn files are the 2D maps used for navigation. Each pixel is a distance of 1cm.
2. The .yaml files can be used for loading the maps into ROS, by using the map_server package.
3. The .txt files ending in "tasks" contain navigation episode with distance (0,4]m between the 
   start and the goal location. The first line contains the start locations, and the second line
   contains the corresponding goal locations.
4. The .txt files ending in "episodes" contain randomly sampled start and goal locations, with
   intermediate goals at a distance of 1m. The first location is the start, each subsequent is 
   the next goal. Each line represents a separate navigation scenario. Our evaluation was done
   on the 10 first episodes.
