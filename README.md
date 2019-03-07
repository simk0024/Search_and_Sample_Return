## Project: Search and Sample Return

---

**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 
[threshed_rock]: ./output/threshed_rock.jpg
[obs_map]: ./output/obs_map.jpg
[view]: ./output/view.jpg
[warped]: ./output/warped.jpg
[threshed]: ./output/threshed.jpg
[rover-centric]: ./output/rover-centric.jpg
[worldmap]: ./output/worldmap.jpg
[output]: ./output/output.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points


### Notebook Analysis

In the virtual world of this project, the color of navigable terrain, obstacle, and rocks are very different. It is rather simple to be diferrentated and identify using color selection/color threshold.

Obstacles are identified by:
```
# create a mask of the field of view of rover
mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))

# create the obstacles map (area that's not navigable terrain is obstacle)
obs_map = np.absolute(np.float32(threshed) - 1) * mask
```

Rocks are identified by:
```
# create a find rock function that select the color of rock (yellow)
def find_rocks(img, rgb_thresh=(110, 110, 50)):
    rock_select = np.zeros_like(img[:, :, 0])
    rock_thresh = (img[:, :, 0] > rgb_thresh[0])\
                & (img[:, :, 1] > rgb_thresh[1])\
                & (img[:, :, 2] < rgb_thresh[2])
    rock_select[rock_thresh] = 1
    return rock_select
```
Robot view 		    			|Color selection
:------------------:			|:------------------------:
![alt text][image3] *Rock image*|![alt text][threshed_rock]*Rock selection *
![alt text][image2] *Obstacle img*|![alt tect][obs_map]*Obstacle selection (perspective transformed)*



#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

`process_image(img)` function process image and output a 3 in 1 image that includes rover view, perspetive transform view, and worldmap.

In the `process_image(img)` function, `img` is first be apllied `perspect_transform()`, where source and destination points are pre-defined. After perspective transform is applied, we get a `warped` image and a `mask`, `mask` is show the field of view of rover.

Then,
1. apply `color_thresh()` on the `warped` image to identify navigable terrain. 
2. overlap `warped` with `mask` to identify obstacles.
3. apply `find_rocks()` on `warped` to identify rocks. If rocks found, result will be further processed to locate it on worldmap

Once we have the thresholded image `nav_map`, convert image coordinates to rover-centric coordinates with `rover_coords()` function. 

Finally, convert the rover-centric pixel values to world coordinates with `pix_to_world()` function. the world coordinates are used to be plotted in the `worldmap`.

In the `worldmap`, after image is processed, navigable terrain, obstacles and rocks will be marked. Navigable terrain is marked in blue color, obstacles are marked in red and rocks are marked in white.

|`process_image()` steps' result|
|:---------:|
|![][view]*`view`*|![][warped]*`warped`*|![][threshed]*`threshed`*
|![][rover-centric]*rover-centric pixel values*|![][worldmap]*worldmap*|![][output]*output of `process_image()`*

```
def process_image(img):
    # TODO: 
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1] / 2 - dst_size, img.shape[0] - bottom_offset], \
                              [img.shape[1] / 2 + dst_size, img.shape[0] - bottom_offset], \
                              [img.shape[1] / 2 + dst_size, img.shape[0] - bottom_offset - 2 * dst_size],\
                              [img.shape[1] / 2 - dst_size, img.shape[0] - bottom_offset - 2 * dst_size]])
    
    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_map = color_thresh(warped)
    obs_map = np.absolute(np.float32(nav_map) - 1) * mask
    rock_found = find_rocks(warped)
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    xpix, ypix = rover_coords(nav_map) 
    #dist, angles = to_polar_coords(xpix, ypix)
    
    # 5) Convert rover-centric pixel values to world coords
    world_size = data.worldmap.shape[0]
    scale = 2 * dst_size
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]
    nav_xpix_world, nav_ypix_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
    
    obs_map = np.absolute(np.float32(threshed) - 1) * mask
    obs_xpix, obs_ypix = rover_coords(obs_map)
    obs_xpix_world, obs_ypix_world = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)
    
    # 6) Update worldmap (to be displayed on right side of screen)
        # Example: data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          data.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          data.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    data.worldmap[obs_ypix_world, obs_xpix_world, 0] = 255
    data.worldmap[nav_ypix_world, nav_xpix_world, 2] = 255
    
    if rock_found.any():
        rock_xpix, rock_ypix = rover_coords(rock_found)
        rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
        data.worldmap[rock_y_world, rock_x_world, :] = 255

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped, mask = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

`perception_step()` is quite similar to process image, the differences is instead of read the image and data from recorded datasheet, the image and data is read from Rover in realtime, and feedback to Rover. The feedback includes the navigation angle nad distance (`nav_dists`, `nav_angles`), to inform Rover where to move to.

```
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    img = Rover.img 
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1] / 2 - dst_size, img.shape[0] - bottom_offset], \
                              [img.shape[1] / 2 + dst_size, img.shape[0] - bottom_offset], \
                              [img.shape[1] / 2 + dst_size, img.shape[0] - bottom_offset - 2 * dst_size],\
                              [img.shape[1] / 2 - dst_size, img.shape[0] - bottom_offset - 2 * dst_size]])
							  
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
	
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_img = color_thresh(warped)
	
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = nav_img * 255
	
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(nav_img)
	
    # 6) Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    nav_x_world, nav_y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    obs_img = np.absolute(np.float32(nav_img)-1) * mask
    obs_xpix, obs_ypix = rover_coords(obs_img)
    obs_x_world, obs_y_world = pix_to_world(obs_xpix, obs_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    	
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[obs_y_world, obs_x_world, 0] += 100
    Rover.worldmap[nav_y_world, nav_x_world, 2] += 10
	
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    dists, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dists
    Rover.nav_angles = angles

    rock_img = find_rocks(warped)
    if rock_img.any():
        rock_x, rock_y = rover_coords(rock_img)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]

        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        Rover.vision_image[:, :, 1] = rock_img * 255
    else:
        Rover.vision_image[:, :, 1] = 0    
    
    return Rover
```

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

I am running simulator for autonomous mode in following setting:
Screen resolution: 1280x 720
Graphics quality: Good

When running the autonomous mode with my `drive_rover.py`, rover might be trapped in a infinite loop in some corner and couldn't get out from the situation without manual interrupt. Thus, adding a discovered_flag and path planning to make rover will try go to undiscovered area when trapped.



