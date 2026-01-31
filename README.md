🌀 Maze Generator Pro - Ultimate Blender Addon

Version: 4.5

Compatibility: Blender 2.80 - 4.x

Category: Add Mesh / Level Design

📖 Introduction

Maze Generator Pro is a powerful, professional-grade tool designed for game developers, 3D artists, and level designers. It allows you to generate complex, fully-featured 3D mazes in seconds. Unlike simple grid generators, this tool creates game-ready levels complete with physics, lighting, rooms, loot, and enemies.

📥 How to Install

Download the Script:

Save the provided python code as a file named maze_addon.py.

Open Blender:

Go to Edit > Preferences.

Install Add-on:

Select the Add-ons tab on the left.

Click the "Install..." button at the top right.

Navigate to where you saved maze_addon.py and select it.

Enable:

Search for "Maze" in the add-on list.

Check the box next to "Add Mesh: Maze Generator Pro" to enable it.

🚀 How to Run

In the 3D Viewport, press the "N" key on your keyboard to open the Sidebar (the menu on the right side).

Click on the tab named "Maze Pro".

You will see the full control panel.

Click the big "Generate Maze" button at the bottom to create your first maze!

🎛️ Features & Settings Guide

1. Core Settings

Algorithm: Choose the mathematical logic for the maze layout.

Recursive Backtracker: Long, winding paths. Hardest to solve.

Hunt-and-Kill: Similar to backtracker but with different twisting patterns.

Prim's / Kruskal's: More randomized, creating short dead ends and a more "organic" feel.

Recursive Division: Creates large rectangular rooms and long straight corridors.

Sidewinder / Eller's: Good for infinite-like mazes or row-by-row generation.

Theme: (Visual Preset) - Currently affects internal logic, expandable for future asset packs.

Width / Height: Sets the grid size of the maze (e.g., 25x25 blocks).

Cell Size: The physical size of each grid block in Blender units (meters).

2. Geometry & Style

Wall Height: How tall the walls are.

Wall Thickness: Controls how thick the walls are relative to the path. Lower values create thin partitions; higher values create blocky pillars.

Height Variance (Jitter): Adds randomness to wall height. Use this to create "ruined" or "ancient" walls that aren't perfectly flat.

Brick Texture: (Checkbox) Automatically applies a procedural Red Brick material with mortar and 3D bump mapping.

Brick Scale: Controls the size of the bricks on the walls.

Add Ceiling: Generates a roof mesh on top of the walls (useful for indoor dungeon levels).

Auto UV Unwrap: Automatically unwraps the UVs of walls, floors, and ceilings using "Cube Projection" so textures fit perfectly.

3. Gameplay & Layout

This section turns a simple maze into a playable Game Level.

Remove Dead Ends (Braid Chance): * 0.0: A "Perfect Maze" (only one solution, lots of dead ends).

0.1 - 1.0: Removes walls to create loops. Essential for FPS/RPG games so players don't get stuck in dead ends constantly.

Add Rooms: Carves out open spaces inside the maze. Great for boss arenas or loot rooms.

Smart Placement: Rooms will never overwrite the "Solution Path", ensuring the maze is always solvable.

Doors: Each room has exactly one entry/exit point connecting it to the maze.

Min/Max Room Size: Controls how big the generated rooms can be.

4. Extras (Loot & Markers)

Loot Count: Randomly spawns "Treasure" objects (Gold Pyramids) throughout the maze paths.

Show Markers: Adds a Green Pillar at the Start and a Red Pillar at the Finish line.

Show Solution: Draws a visible red line showing the shortest path from Start to End. Great for debugging or creating "hint" trails.

5. Atmosphere & Physics (Game Ready)

Add Lights: Places Point Lights (Torches) at intersections and dead ends to light up the maze.

Light Color: Change the color (default is Fire Orange).

Light Power: Intensity of the lights.

Add Physics: * Adds Passive Rigid Body physics to Walls and Floors (so objects collide with them).

Adds Active Rigid Body to Bosses and Loot (so they can be knocked over).

6. Seed (Randomness)

Randomize Seed: If checked, every click creates a unique maze.

Seed Number: If unchecked, you can enter a specific number (e.g., 1234). This allows you to recreate the exact same maze layout again if you need to modify settings.

🛠️ Step-by-Step Workflow Example

Goal: Create a Dungeon Level for a game.

Set Width/Height to 31 for a medium map.

Choose Algorithm: Recursive Backtracker for long corridors.

Set Wall Height to 3.0 meters.

Enable Brick Texture and set scale to 4.0.

Enable Add Ceiling to make it an indoor dungeon.

Set Remove Dead Ends to 0.5 (50%) to add loops for better gameplay flow.

Set Add Rooms to 3 to create boss arenas.

Enable Add Lights and set color to a dim orange.

Enable Add Physics so your game character can walk inside immediately.

Click Generate Maze.

⚠️ Troubleshooting

"Texture looks stretched": Make sure "Auto UV Unwrap" is checked. The script uses Cube Projection to fix this.

"Rooms are missing": If the maze is too small (e.g., Width 5), large rooms might not fit. Increase the Maze Width/Height or decrease Room Size.

"My character falls through the floor": Ensure "Add Physics" is checked, or manually add collision to the "MazeFloor" object in Blender.

Enjoy building!