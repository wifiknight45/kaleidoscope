# kaleidoscope
Pygame + NumPy to generate psychedelic textures in real time

Kaleidoscope
A real-time interactive kaleidoscope built with Pygame and NumPy. Ten pattern generators produce continuously evolving visuals that are mirrored and rotated into a kaleidoscope effect. An auto-cycle mode rotates through all patterns automatically.

Requirements
Python 3.9 or higher is required along with two dependencies.

pip install pygame numpy

Running
python3 kaleidoscope.py

Controls
SPACE          cycle to the next pattern
A              toggle auto-cycle mode
F              toggle fade transition between patterns
UP / DOWN      increase or decrease rotation speed
RIGHT / LEFT   add or remove mirror slices
S              save a screenshot to the screenshots folder
ESC or Q       quit
Patterns
Plasma         psychedelic sine-noise field
Rainbow        smooth rolling hue gradient
Splatter       neon paint blobs
Swirl          concentric radial ripples
Stripes        shifting neon lattice
Mandala        polar rose petal geometry
Lava           molten colour blobs
Starfield      zooming star tunnel
Crystals       Voronoi crystal shards with glowing edges
Aurora         northern lights curtain bands

Configuration
At the top of kaleidoscope.py a small block of constants controls default behaviour.
WIDTH, HEIGHT      window size in pixels, default 800 by 800
SLICES             number of mirror segments, default 12
ROT_SPEED          rotation speed in degrees per frame, default 0.5
AUTO_CYCLE_SECS    seconds spent on each pattern in auto mode, default 6
FADE_FRAMES        number of frames used for the blend transition, default 45
SCREENSHOT_DIR     folder where screenshots are saved, default screenshots
Screenshots
Screenshots are saved as PNG files in the screenshots folder, which is created automatically on first save. Files are named kaleid_0000.png, kaleid_0001.png and so on.
