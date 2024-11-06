import sys
import time

def loading_animation():
    bunny0 = [
        "       ",
        "        ^ ^",
        "      ( *.*)",
        "     o( u u)"
    ]

    bunny1 = [
        "        ^ ^",
        "      ( *.*)",
        "     o( u u)",
        "       "
    ]

    hop = True
    
    ground = "----^--------------------^^-----------------^-----------^^-----^-----------^^-------------^--------^^----"  # The "ground" under the bunny
    ground_length = len(ground)
    num_spaces = 0  # Control the movement of the ground

    while True:
        sys.stdout.write("\033c")  # Clear screen (for most terminals)

        if hop: 
            # Print the bunny, which stays fixed in place
            print(bunny0[0])
            print(bunny0[1])
            print(bunny0[2])
            print(bunny0[3])
        elif not(hop):
            print(bunny1[0])
            print(bunny1[1])
            print(bunny1[2])
            print(bunny1[3])
        
        # Create an effect of infinite ground by shifting
        # The ground moves below the bunny
        print(f"{ground[num_spaces % ground_length:] + ground[:num_spaces % ground_length]}")
        
        # Move the ground "left" (by incrementing num_spaces)
        num_spaces += 1

        # Flip hope to bounce
        hop = not(hop)
        
        # Sleep to slow down the animation
        time.sleep(0.25)
