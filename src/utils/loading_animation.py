import sys
import time

def loading_animation():
    bunny = [
        " ^ ^",
        "( *.*)",
        "( u u)o"
    ]
    
    ground = "----v--------------------vv-------------------v-----------vv-----v-----------vv-------------v--------vv----"  # The "ground" under the bunny
    ground_length = len(ground)
    num_spaces = 0  # Control the movement of the ground

    while True:
        sys.stdout.write("\033c")  # Clear screen (for most terminals)
        
        # Print the bunny, which stays fixed in place
        print(bunny[0])
        print(bunny[1])
        print(bunny[2])
        
        # Create an effect of infinite ground by shifting
        # The ground moves below the bunny
        print(f"{ground[num_spaces % ground_length:] + ground[:num_spaces % ground_length]}")
        
        # Move the ground "left" (by incrementing num_spaces)
        num_spaces += 1
        
        # Sleep to slow down the animation
        time.sleep(0.1)