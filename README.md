> ## Project 1 Writeup: Search and Sample Return 
>> **Date:** May 28, 2018
>> **By:** Paul Griz
>> Project Submission for the Udacity Robotics Software Engineer Nanodegree

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

---



## Autonomous Navigation and Mapping 

### 1. The ``perception_step()``:

1. The ``perspective_transform()`` function: Used to convert the Rover camera's POV to a "top-down" world view. To improve the world map's fidelity, a mask is applied to the transformed perspective. The mask is a slice of the input image taking off 35 pixels from the top and 80 pixels from both left & right sides. The ``cv2.copyMakeBorder()`` function was then applied to the splice in order to match original shape of ``(160, 320)``.  

> The code:

```python
def perspective_transform(img, src, dst):
    # Get transform matrix using cv2.getPerspectiveTransform()
    transform_matrix = cv2.getPerspectiveTransform(src, dst)

    # Warp image using cv2.warpPerspective()
    # Note: warped image has the same size as input image
    warped = cv2.warpPerspective(
        img, transform_matrix, (img.shape[1], img.shape[0]))

    # Added Mask to only process data from the rover's POV
    # [Removes data NOT in Rover's POV]
    mask = cv2.warpPerspective(src=np.ones_like(img[:, :, 0]), M=transform_matrix,
                               dsize=(img.shape[1], img.shape[0]),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Cropped the mask to narrow the Rover's POV. Improves Rover's navigation.
    mask = mask[35:160, 80:240]
    # Adding a black border to cropped mask to regain original shape of (160, 320)
    mask = cv2.copyMakeBorder(
        mask, 35, 0, 80, 80, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return warped, mask
```

> The Output:

![Perspective_Transform_Output](.\images\perspective_transform_output.PNG)

The above output is used in mapping the roughly 200 x 200 meters world map.

![The World Map](.\images\World-Map.png)

Clearing opposing data [Navigable terrain VS. Obstacles] greatly improved fidelity. 

```python
# World Map's Navigable Pixels
Rover.worldmap[y_world, x_world, 2] += 255
# Clear opposing data to improve Fidelity
Rover.worldmap[obs_y_world, obs_x_world, 2] -= 40

# World Map's Obstacles
Rover.worldmap[obs_y_world, obs_x_world, 0] += 255
# Clear opposing data to improve Fidelity
Rover.worldmap[y_world, x_world, 0] -= 90
```

By applying the mask and clearing the opposing data, the rover consistently achieves ~80-90% Fidelity. 

![Fidelity Improvements Results](.\images\Fidelity-Results.PNG)

2. The ``find_rocks()`` function: Used to find rocks by applying a RGB threshold to detect rock samples within the input threshold. Very similar to the ``color_thresh()`` function.

> The code:

```python
def find_rocks(img, levels=(110, 110, 50)):
    # The rocks have high Red and Green levels & low Blue levels
    rockpix = ((img[:, :, 0] > levels[0])
               & (img[:, :, 1] > levels[1])
               & (img[:, :, 2] < levels[2]))

    color_select = np.zeros_like(img[:, :, 0])
    color_select[rockpix] = 1

    return color_select
```

> The output:

![Find_Rocks_Function_Output](.\images\Find_Rocks_Function_Output.PNG)

Once a rock has been located by the Rover, the rock is then added to the world map:

```python
if rock_map.any():
        # At Rock to World Map
        rock_x_world, rock_y_world = pix_to_world(
            rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 255
        Rover.vision_image[:, :, 1] = rock_map * 225
    else:
        Rover.vision_image[:, :, 1] = 0
```

Finally, the Rover's steering is set towards the rock:

```python
if len(rock_dist) > 0:
        if Rover.mode == 'stuck':
            Rover.mode = 'stuck'
        else:
            Rover.mode = 'going_to_rock'
            Rover.rock_dists = rock_dist
            Rover.rock_angle = rock_angles
    else:
        Rover.nav_dists = dist
        Rover.nav_angles = angles
```

---



## 2. `decision_step()`

In ``decision.py``, the ``decision_step( )`` function was utilized as the script's main function. For decision making, the Rover was given six different modes to switch between: [Forward, Reverse, Stop, Cut Out, Going to Rock, and Picking Rock]. 

Each mode has its own function where the action is preformed. The ``decision_step()`` function handles the calling of all functions.

> The code:

```python
def decision_step(Rover):
    # Verify if Rover has vision data
    if Rover.nav_angles is not None:
        # Check for Rover.mode status

        # ---> Reverse
        #====================
        if Rover.mode == 'reverse':
            set_reverse(Rover)
        
        # ---> Stop
        #====================
        elif Rover.mode == 'stop':
            if Rover.stuck_count >= 50:
                Rover.mode = 'reverse'
                return Rover
            set_stop(Rover)
        
        # ---> Forward
        #====================
        elif Rover.mode == 'forward':
            # Setting brake to 0 in case pervious
            # mode applied a brake
            Rover.brake = 0

            # Conditions for Cutting Out:
            if Rover.vel >= 1.3:
                # Call the cut_out() function
                if Rover.cut_out_count >= 50.0:
                    Rover.mode = 'cut_out'
                    return Rover
                # If going Forward, Vel > 1.8 AND Steering
                # at -15 or 15: add to the counter
                elif Rover.steer == 15.0 or Rover.steer == -15.0:
                    Rover.cut_out_count += 1
                # Else: subtract from counter
                else:
                    if Rover.cut_out_count >= 1:
                        Rover.cut_out_count -= 1

            # Checking if Rover is Stuck:
            if Rover.throttle == Rover.throttle_set:
                # If Rover is stuck
                if Rover.stuck_count >= 55.0:
                    Rover.mode = 'reverse'
                    return Rover
                # If going Forward, with Vel in range(-0.2, 0.06)
                # AND in full throttle: add to the counter
                elif Rover.vel < 0.06 and Rover.vel > -0.2:
                    Rover.stuck_count += 1
                # Else: subtract from counter
                else:
                    if Rover.stuck_count >= 0.5:
                        Rover.stuck_count -= 0.5

            # If not stuck OR about to cut out, go forward
            set_forward(Rover)

        # ---> Going to Rock
        #====================
        elif Rover.mode == 'going_to_rock':
            # Checking if Rover is Stuck:
            if Rover.throttle == 0.2 and Rover.near_sample == 0:
                if Rover.stuck_count >= 60.0:
                    Rover.mode = 'reverse'
                    return Rover
                elif Rover.vel < 0.05 and Rover.vel > -0.2:
                    Rover.stuck_count += 1
                else:
                    if Rover.stuck_count >= 0.5:
                        Rover.stuck_count -= 0.5
            going_to_rock(Rover)

        # ---> Picking Rock
        #====================
        elif Rover.mode == 'picking_rock':
            picking_rock(Rover)

        # ---> Cut Out
        #====================
        elif Rover.mode == 'cut_out':
            cut_out(Rover)

        else:
            print("ERROR: Unknown state")

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    if Rover.near_sample == 1 and Rover.vel == 0 and not Rover.picking_up:
        Rover.stuck_count = 0
        Rover.send_pickup = True
    else:
        Rover.send_pickup = False
    return Rover
```

All functions called within the ``decision_step( )`` function preform the required actions and return the Rover. 

### Navigation Improvements

1. The ``cut_out()`` function: Cuts the Rover out of long turns. The list: `Rover.steer_cuts[]`is unevenly distributed to give a randomized approach. These random cuts allow the Rover to eventual navigate the entire map.

> The code:

```python
def cut_out(Rover):
    Rover.mode = 'cut_out'

    # Prevent Rover from turning into an obstacle
    # when cutting out of a turn
    if len(Rover.nav_angles) < Rover.stop_forward:
        Rover.mode = 'stop'
        return Rover

    # Rover.steer_cuts is a list of negative & positive
    # values. The list is used to give the Rover a
    # randomized directional choice. Prevents the Rover
    # from missing areas of the map & infinite circles.
    if Rover.steer_cut_index >= len(Rover.steer_cuts):
        Rover.steer_cut_index = 0
    Rover.steer = Rover.steer_cuts[Rover.steer_cut_index]

    if Rover.cut_out_count >= 1.0:
        Rover.cut_out_count -= 1.0
    else:
        Rover.cut_out_count = 0
        Rover.steer_cut_index += 1
        Rover.mode = 'forward'
        return Rover
```

2. The ``set_reverse()`` function: Called when the ``Rover.stuck_count`` reaches values set within the ``forward`` & ``going_to_rock`` Rover modes. Additionally, used to backup the Rover after picking up a rock. Also, the ``Rover.stuck_in_stuck_counter`` decreased the chances of the Rover getting stuck on top of or within obstacles.  

> The code:

```python
def set_reverse(Rover):
    Rover.mode = 'reverse'
    Rover.brake = 0
    Rover.throttle = -0.6

    # Prevents the Rover from getting stuck on top of
    # or within rocks.
    if Rover.vel > -0.02 and Rover.vel <= 0.02:
        Rover.stuck_in_stuck_counter += 1
    else:
        if Rover.stuck_in_stuck_counter >= 0.5:
            Rover.stuck_in_stuck_counter -= 0.5

    # Required to prevent a Numpy RuntimeWarning
    # If the rover is in front of an obstacle
    # np.mean() cannot comput len(Rover.nav_angles)
    if len(Rover.nav_angles) < Rover.go_forward:
        Rover.steer = 0
    elif Rover.stuck_in_stuck_counter >= 25:
        Rover.steer = 15
        Rover.throttle = 0
    else:
        # Steer in the opposite direction as direction
        # that lead to getting stuck
        Rover.steer = -(np.clip(
            np.mean(Rover.nav_angles * 180/np.pi), -15, 15))

    if Rover.stuck_count >= 0.5:
        Rover.stuck_count -= 0.5
    else:
        Rover.throttle = Rover.throttle_set
        Rover.stuck_count = 0
        Rover.stuck_in_stuck_counter = 0
        Rover.mode = 'forward'
        return Rover
```

3. The ``going_to_rock()`` function: Called when ``Rover.mode = 'going_to_rock'`` is set within ``perception.py`` if a rock is visible. Sets the Rover's steering toward the closest rock.

> The code:

```python
def going_to_rock(Rover):
    Rover.mode = 'going_to_rock'

    # Pointing steer angles to the closest Rock
    Rover.steer = np.clip(
        np.mean(Rover.rock_angle * 180/np.pi), -15, 15)

    # Slow Down & Prevent backwards movement
    if Rover.vel > 1 or Rover.vel < -0.03:
        Rover.brake = 1
    else:
        Rover.brake = 0

    # Setting a low max velocity to prevent
    # hard stops when near a sample
    if Rover.vel < 0.8:
        Rover.throttle = 0.2
    else:
        Rover.throttle = 0

    # If the Rover is close enough to pick-up
    if Rover.near_sample == 1:
        Rover.brake = 10
        Rover.mode = 'picking_rock'
        return Rover
```

4. In the ``picking_rock()`` function, doing a short reverse after picking up a rock and steering in the opposite direction really improved navigation times.  Prevents the Rover from turning around after picking a rock up against a wall. 

> The code:

```python
def picking_rock(Rover):
    """Called when Rover.near_sample == 1"""
    Rover.mode = 'picking_rock'

    Rover.steer = 0

    if not Rover.picking_up:
        Rover.send_pickup = True
    else:
        Rover.send_pickup = False

    if Rover.near_sample == 0:
        # Do a short backup after picking rock
        # Prevents Rover from turning around
        # after picking a rock near a wall
        Rover.stuck_count = 30
        Rover.mode = 'reverse'
        return Rover
```

### Debugging Help

In ``supporting_functions.py``: A simple console application was made to keep track of the Rover's data points. ``Rover.console_log_counter >= 10.0`` was used to reduce the number of times per second the console was refreshed to prevent wasting system resources. 

> The code:

```python
if Rover.console_log_counter >= 10.0:
    # Print out the fields in the telemetry data dictionary
    os.system('cls')
    print("""\n{14}
            \nTotal Time = {0:.2f} \t FPS = {16}
            \nSpeed = {1:.2f} \t Position = {2}
            \nThrottle = {3} \t steer_angle = {4:.2f}
            \n{14}
            \nsamples collected = {5} \t samples remaining = {6}
            \nnear_sample = {7} \t sending pickup = {8} 
            \npicking_up = {9}
            \n{14}
            \nCurrent Mode = {10}
            \n{14}
            \nStuck_Count = {11}
            \nStuck in Stuck Count = {12}
            \n{14}
            \nCut Out Count = {13}
            \nCut Out Index = {15}
            \n{14}"""
          .format(Rover.total_time, Rover.vel, Rover.pos, Rover.throttle, Rover.steer,
                  Rover.samples_collected, Rover.samples_to_find, Rover.near_sample,
                  Rover.send_pickup, Rover.picking_up, Rover.mode, Rover.stuck_count,
                  Rover.stuck_in_stuck_counter, Rover.cut_out_count, bar_string, 
                  Rover.steer_cut_index, Rover.fps))
    Rover.console_log_counter = 0.0
```

> Result:

![Console-Example](.\images\Console-Example.PNG)

---



## Results

| Data Point         | Average         |
| ------------------ | --------------- |
| Mapped             | ~87-95%         |
| Fidelity           | ~81-85%         |
| Located Rocks      | 5-6             |
| Collected Rocks    | 5-6             |
| Time till Complete | 450-750 Seconds |

### Settings:

| Setting           | Value      |
| ----------------- | ---------- |
| Screen Resolution | 1280 x 800 |
| Graphics Quality  | Good       |
| Average FPS       | 46-55      |



---



## Possible Improvements

1. Adding methods to prevent the Rover from going over the same area again. I chose to randomize the Rover's navigation which does eventual cover the entire map. However, if time or fuel were constraints, my method would not be the most efficient.
2. The Rover can find and pick up all the Rock samples from the simulation. However, adding a drop off spot in the middle of the map would be required for real practice use.
3. Adding methods for changing the ``Rover.max_vel`` would improve navigation times. For example, allowing the Rover to accelerate over the ``Rover.max_vel`` if going down a straight path without any near obstacles.

----

