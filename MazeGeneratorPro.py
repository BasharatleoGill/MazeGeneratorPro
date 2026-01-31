bl_info = {
    "name": "Maze Generator Pro",
    "author": "Your Name",
    "version": (4, 6),
    "blender": (2, 80, 0),
    "location": "View3D > N-Panel > Maze Pro",
    "description": "Professional 3D Maze Generator with Physics & Lighting",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh",
}

import bpy
import random
import sys
import math
from mathutils import Vector

# Increase recursion depth for complex algorithms
sys.setrecursionlimit(3000)

# ==========================================
# 1. SETTINGS (PROPERTIES)
# ==========================================
class MazeProperties(bpy.types.PropertyGroup):
    # --- CORE ---
    maze_algorithm: bpy.props.EnumProperty(
        name="Algorithm",
        description="Choose the maze generation logic",
        items=[
            ('BACKTRACKER', "Recursive Backtracker", "Long winding paths, high difficulty"),
            ('HUNT_KILL', "Hunt-and-Kill", "Many dead ends, similar to Backtracker"),
            ('PRIMS', "Prim's Algorithm", "Radiates from center, short paths"),
            ('KRUSKAL', "Kruskal's Algorithm", "Randomized connections, varied texture"),
            ('RECURSIVE_DIVISION', "Recursive Division", "Open areas with long walls"),
            ('SIDEWINDER', "Sidewinder", "Bottom-to-top flow, horizontal runs"),
            ('ELLER', "Eller's Algorithm", "Row by row generation (infinite logic)"),
        ],
        default='BACKTRACKER'
    )
    
    # --- THEMES ---
    maze_theme: bpy.props.EnumProperty(
        name="Theme",
        description="Visual style of the maze",
        items=[
            ('DEFAULT', "Standard Grey", "Basic prototyping look"),
            ('DUNGEON', "Dark Dungeon", "Rough stone, dark atmosphere"),
            ('SCIFI', "Sci-Fi Base", "Metallic, sleek, neon hints"),
            ('GARDEN', "Hedge Maze", "Green organic look"),
            ('ICE', "Frozen Cave", "Blueish, shiny ice"),
        ],
        default='DEFAULT'
    )

    maze_width: bpy.props.IntProperty(
        name="Width",
        description="Width of the maze area",
        default=25,
        min=5,
        soft_max=100
    )
    
    maze_height: bpy.props.IntProperty(
        name="Height",
        description="Length of the maze area",
        default=25,
        min=5,
        soft_max=100
    )
    
    cell_size: bpy.props.FloatProperty(
        name="Cell Size",
        description="Size of each block",
        default=1.0,
        min=0.1
    )

    # --- GEOMETRY & STYLE ---
    wall_height: bpy.props.FloatProperty(
        name="Wall Height",
        description="Height of the maze walls",
        default=2.0,
        min=0.1
    )
    
    wall_thickness: bpy.props.FloatProperty(
        name="Wall Thickness",
        description="Thickness relative to cell size (1.0 = Full Block)",
        default=1.0,
        min=0.1,
        max=1.0,
        subtype='FACTOR'
    )
    
    wall_jitter: bpy.props.FloatProperty(
        name="Height Variance",
        description="Randomize wall heights for a ruined/organic look",
        default=0.0,
        min=0.0,
        max=2.0
    )
    
    use_brick_wall: bpy.props.BoolProperty(
        name="Brick Texture",
        description="Apply procedural brick material to walls",
        default=True
    )
    
    brick_scale: bpy.props.FloatProperty(
        name="Brick Scale",
        description="Size of the bricks (Higher = Smaller bricks)",
        default=5.0,
        min=1.0,
        max=50.0
    )
    
    add_ceiling: bpy.props.BoolProperty(
        name="Add Ceiling",
        description="Generate a ceiling mesh on top of the walls",
        default=False
    )
    
    auto_uv: bpy.props.BoolProperty(
        name="Auto UV Unwrap",
        description="Automatically unwrap UVs for texturing",
        default=True
    )

    # --- GAMEPLAY & LAYOUT ---
    braid_chance: bpy.props.FloatProperty(
        name="Remove Dead Ends",
        description="Chance to open dead ends to create loops (0 = Perfect Maze, 1 = Very Open)",
        default=0.0,
        min=0.0,
        max=1.0
    )
    
    room_count: bpy.props.IntProperty(
        name="Add Rooms",
        description="Carve out open rooms in the maze",
        default=0,
        min=0,
        max=20
    )
    
    room_size_min: bpy.props.IntProperty(
        name="Min Room Size",
        description="Minimum size of rooms (grid units)",
        default=2,
        min=2,
        max=10
    )
    
    room_size_max: bpy.props.IntProperty(
        name="Max Room Size",
        description="Maximum size of rooms (grid units)",
        default=4,
        min=2,
        max=10
    )

    loot_count: bpy.props.IntProperty(
        name="Loot Count",
        description="Number of treasure items to drop in the maze",
        default=5,
        min=0
    )
    
    show_markers: bpy.props.BoolProperty(
        name="Start/End Markers",
        description="Place pillars at Start and End points",
        default=True
    )

    show_solution: bpy.props.BoolProperty(
        name="Show Solution",
        description="Draw a line showing the correct path from Start to Finish",
        default=False
    )
    
    # --- ATMOSPHERE & PHYSICS ---
    add_lights: bpy.props.BoolProperty(
        name="Add Lights",
        description="Place point lights throughout the maze",
        default=False
    )
    
    light_color: bpy.props.FloatVectorProperty(
        name="Light Color",
        subtype='COLOR',
        default=(1.0, 0.6, 0.2), # Orange Torch
        min=0.0, max=1.0
    )
    
    light_power: bpy.props.FloatProperty(
        name="Light Power",
        default=50.0,
        min=1.0, max=1000.0
    )
    
    add_physics: bpy.props.BoolProperty(
        name="Add Physics",
        description="Add Passive Rigid Body to Walls & Floor for collision",
        default=False
    )
    
    # --- SEED ---
    use_random_seed: bpy.props.BoolProperty(
        name="Randomize Seed",
        description="Generate a random maze every time",
        default=False
    )

    seed: bpy.props.IntProperty(
        name="Seed",
        description="Change this number to generate a different maze pattern",
        default=0,
        min=0
    )

# ==========================================
# 2. ALGORITHMS & HELPERS
# ==========================================

def create_brick_material(name, scale):
    """Creates a professional procedural brick material with noise/grunge"""
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # --- NODE SETUP ---
    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-1000, 0)
    
    brick = nodes.new('ShaderNodeTexBrick')
    brick.location = (-600, 200)
    brick.offset = 0.5
    brick.inputs['Scale'].default_value = scale
    brick.inputs['Color1'].default_value = (0.5, 0.15, 0.1, 1.0) 
    brick.inputs['Color2'].default_value = (0.35, 0.1, 0.05, 1.0)
    brick.inputs['Mortar'].default_value = (0.15, 0.15, 0.15, 1.0)
    brick.inputs['Mortar Size'].default_value = 0.02
    
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-600, -200)
    noise.inputs['Scale'].default_value = 35.0
    noise.inputs['Detail'].default_value = 15.0
    noise.inputs['Roughness'].default_value = 0.7
    
    mix_color = nodes.new('ShaderNodeMixRGB')
    mix_color.location = (-300, 200)
    mix_color.blend_type = 'MULTIPLY'
    mix_color.inputs['Fac'].default_value = 0.35
    
    bump_grain = nodes.new('ShaderNodeBump')
    bump_grain.location = (-300, -100)
    bump_grain.inputs['Strength'].default_value = 0.2
    
    bump_shape = nodes.new('ShaderNodeBump')
    bump_shape.location = (0, -100)
    bump_shape.inputs['Strength'].default_value = 1.0
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    
    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (600, 0)
    
    # --- LINKING ---
    links.new(coord.outputs['UV'], brick.inputs['Vector'])
    links.new(coord.outputs['UV'], noise.inputs['Vector'])
    links.new(brick.outputs['Color'], mix_color.inputs[1])
    links.new(noise.outputs['Color'], mix_color.inputs[2])
    links.new(mix_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(noise.outputs['Fac'], bsdf.inputs['Roughness'])
    links.new(noise.outputs['Fac'], bump_grain.inputs['Height'])
    links.new(bump_grain.outputs['Normal'], bump_shape.inputs['Normal'])
    links.new(brick.outputs['Fac'], bump_shape.inputs['Height'])
    links.new(bump_shape.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    return mat

def solve_bfs_path(grid, w, h, start, end):
    queue = [[start]]
    visited = {start}
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == end: return set(path)
        cx, cy = node
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    queue.append(new_path)
    return set()

def algo_recursive_backtracker(grid, w, h, start_x, start_y, directions):
    grid[start_x][start_y] = 0
    stack = [(start_x, start_y)]
    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < w - 1 and 0 < ny < h - 1:
                if grid[nx][ny] == 1: neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            grid[cx + dx // 2][cy + dy // 2] = 0
            grid[nx][ny] = 0
            stack.append((nx, ny))
        else: stack.pop()

def algo_hunt_and_kill(grid, w, h, start_x, start_y, directions):
    grid[start_x][start_y] = 0
    current_x, current_y = start_x, start_y
    while current_x:
        while True:
            neighbors = []
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 < nx < w - 1 and 0 < ny < h - 1:
                    if grid[nx][ny] == 1: neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                grid[current_x + dx // 2][current_y + dy // 2] = 0
                grid[nx][ny] = 0
                current_x, current_y = nx, ny
            else: break 
        current_x, current_y = None, None
        for x in range(1, w - 1, 2):
            for y in range(1, h - 1, 2):
                if grid[x][y] == 1: 
                    visited_neighbors = []
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 < nx < w - 1 and 0 < ny < h - 1:
                            if grid[nx][ny] == 0: visited_neighbors.append((nx, ny, dx, dy))
                    if visited_neighbors:
                        nx, ny, dx, dy = random.choice(visited_neighbors)
                        grid[x + dx // 2][y + dy // 2] = 0
                        grid[x][y] = 0
                        current_x, current_y = x, y
                        break
            if current_x: break

def algo_prims(grid, w, h, start_x, start_y, directions):
    grid[start_x][start_y] = 0
    frontier = []
    def add_frontier(cx, cy):
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < w - 1 and 0 < ny < h - 1:
                if grid[nx][ny] == 1: frontier.append((nx, ny, cx, cy))
    add_frontier(start_x, start_y)
    while frontier:
        idx = random.randrange(len(frontier))
        nx, ny, px, py = frontier.pop(idx)
        if grid[nx][ny] == 1:
            grid[nx][ny] = 0
            grid[(nx + px) // 2][(ny + py) // 2] = 0
            add_frontier(nx, ny)

def algo_kruskal(grid, w, h):
    sets = {}
    edges = []
    for x in range(1, w - 1, 2):
        for y in range(1, h - 1, 2):
            sets[(x, y)] = (x, y)
            grid[x][y] = 0 
            if x + 2 < w - 1: edges.append((x, y, 2, 0))
            if y + 2 < h - 1: edges.append((x, y, 0, 2))
    random.shuffle(edges)
    def find(item):
        if sets[item] != item: sets[item] = find(sets[item])
        return sets[item]
    def union(set1, set2):
        root1 = find(set1)
        root2 = find(set2)
        if root1 != root2:
            sets[root1] = root2
            return True
        return False
    for x, y, dx, dy in edges:
        nx, ny = x + dx, y + dy
        if union((x, y), (nx, ny)):
            grid[x + dx // 2][y + dy // 2] = 0

def algo_sidewinder(grid, w, h):
    for y in range(1, h - 1, 2):
        run = []
        for x in range(1, w - 1, 2):
            grid[x][y] = 0
            run.append((x, y))
            at_eastern_boundary = (x + 2 >= w - 1)
            at_northern_boundary = (y + 2 >= h - 1)
            should_close = at_eastern_boundary or (not at_northern_boundary and random.choice([True, False]))
            if should_close:
                member = random.choice(run)
                mx, my = member
                if not at_northern_boundary:
                    grid[mx][my+1] = 0
                    grid[mx][my+2] = 0
                run = []
            else: grid[x+1][y] = 0

def algo_eller(grid, w, h):
    row_set = {}
    next_set_id = 0
    for y in range(1, h - 1, 2):
        for x in range(1, w - 1, 2):
            grid[x][y] = 0
            if x not in row_set:
                row_set[x] = next_set_id
                next_set_id += 1     
        for x in range(1, w - 3, 2):
            right_x = x + 2
            if row_set[x] != row_set[right_x]:
                if random.choice([True, False]):
                    grid[x+1][y] = 0
                    target_set = row_set[x]
                    src_set = row_set[right_x]
                    for k in row_set:
                        if row_set[k] == src_set:
                            row_set[k] = target_set
        if y + 2 < h - 1:
            verticals = {}
            for x in range(1, w - 1, 2):
                sid = row_set[x]
                if sid not in verticals: verticals[sid] = []
                verticals[sid].append(x)
            next_row_set = {}
            for sid, cols in verticals.items():
                random.shuffle(cols)
                to_carve = cols[:max(1, random.randint(1, len(cols)))]
                for cx in to_carve:
                    grid[cx][y+1] = 0
                    grid[cx][y+2] = 0
                    next_row_set[cx] = sid
            row_set = next_row_set
    y = ((h - 2) // 2) * 2 + 1
    for x in range(1, w - 3, 2):
        if row_set.get(x) != row_set.get(x+2):
            grid[x+1][y] = 0

def algo_recursive_division(grid, w, h):
    for x in range(w):
        for y in range(h):
            if x == 0 or x == w-1 or y == 0 or y == h-1: grid[x][y] = 1
            else: grid[x][y] = 0
    def divide(bx, by, bw, bh):
        wall_slots_h = (bh - 3) // 2
        wall_slots_w = (bw - 3) // 2
        can_split_h = wall_slots_h > 0
        can_split_w = wall_slots_w > 0
        if not can_split_h and not can_split_w: return
        horizontal = False
        if can_split_h and can_split_w:
            if bw > bh: horizontal = False
            elif bh > bw: horizontal = True
            else: horizontal = random.choice([True, False])
        elif can_split_h: horizontal = True
        else: horizontal = False
        if horizontal:
            if wall_slots_h <= 0: return
            wy = by + 2 + random.randrange(wall_slots_h) * 2
            passage_slots = (bw - 1) // 2
            px = bx + 1 + (random.randrange(passage_slots) * 2 if passage_slots > 0 else 0)
            for x in range(bx, bx + bw): grid[x][wy] = 1 
            grid[px][wy] = 0
            divide(bx, by, bw, wy - by)
            divide(bx, wy, bw, by + bh - wy)
        else:
            if wall_slots_w <= 0: return
            wx = bx + 2 + random.randrange(wall_slots_w) * 2
            passage_slots = (bh - 1) // 2
            py = by + 1 + (random.randrange(passage_slots) * 2 if passage_slots > 0 else 0)
            for y in range(by, by + bh): grid[wx][y] = 1
            grid[wx][py] = 0
            divide(bx, by, wx - bx, bh)
            divide(wx, by, bx + bw - wx, bh)
    divide(0, 0, w, h)

# ==========================================
# 3. POST PROCESSING
# ==========================================
def apply_braiding(grid, w, h, chance):
    dead_ends = []
    for x in range(1, w-1, 2):
        for y in range(1, h-1, 2):
            if grid[x][y] == 0:
                walls = 0
                neighbors = []
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dx, dy in dirs:
                    if grid[x+dx][y+dy] == 1: walls += 1
                    nx, ny = x + dx*2, y + dy*2
                    if 0 < nx < w-1 and 0 < ny < h-1: neighbors.append((dx, dy))
                if walls == 3: dead_ends.append((x, y, neighbors))
    for x, y, neighbors in dead_ends:
        if random.random() < chance and neighbors:
            dx, dy = random.choice(neighbors)
            grid[x + dx][y + dy] = 0 

def apply_rooms_smart(grid, w, h, count, min_s, max_s, protected_cells):
    if min_s > max_s: min_s, max_s = max_s, min_s
    boss_locations = []
    existing_rooms = []
    
    for _ in range(count):
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            attempts += 1
            rw_nodes = random.randint(min_s, max_s)
            rh_nodes = random.randint(min_s, max_s)
            rw_span, rh_span = rw_nodes * 2, rh_nodes * 2
            
            if w - 1 - rw_span <= 1 or h - 1 - rh_span <= 1: continue 
            rx = random.randrange(1, w - 1 - rw_span, 2)
            ry = random.randrange(1, h - 1 - rh_span, 2)
            
            margin = 2
            new_min_x = rx - 1 - margin
            new_max_x = rx + rw_span + 1 + margin
            new_min_y = ry - 1 - margin
            new_max_y = ry + rh_span + 1 + margin
            
            overlap = False
            for (ex_min_x, ex_max_x, ex_min_y, ex_max_y) in existing_rooms:
                if (new_min_x < ex_max_x and new_max_x > ex_min_x and
                    new_min_y < ex_max_y and new_max_y > ex_min_y):
                    overlap = True
                    break
            if overlap: continue

            conflict = False
            for x in range(rx - 1, rx + rw_span + 2):
                for y in range(ry - 1, ry + rh_span + 2):
                    if 0 <= x < w and 0 <= y < h:
                        if (x, y) in protected_cells:
                            conflict = True
                            break
                if conflict: break
            if conflict: continue

            candidates = []
            if ry - 1 > 0:
                for x in range(rx, rx + rw_span + 1, 2):
                    if grid[x][ry - 2] == 0: candidates.append((x, ry - 1))
            if ry + rh_span + 1 < h - 1:
                for x in range(rx, rx + rw_span + 1, 2):
                    if grid[x][ry + rh_span + 2] == 0: candidates.append((x, ry + rh_span + 1))
            if rx - 1 > 0:
                for y in range(ry, ry + rh_span + 1, 2):
                    if grid[rx - 2][y] == 0: candidates.append((rx - 1, y))
            if rx + rw_span + 1 < w - 1:
                for y in range(ry, ry + rh_span + 1, 2):
                    if grid[rx + rw_span + 2][y] == 0: candidates.append((rx + rw_span + 1, y))
            
            if not candidates: continue
            
            placed = True
            existing_rooms.append((rx - 1, rx + rw_span + 1, ry - 1, ry + rh_span + 1))
            
            for i in range(rw_span + 3):
                cx = rx - 1 + i
                if 0 <= cx < w:
                    if ry - 1 >= 0: grid[cx][ry - 1] = 1
                    if ry + rh_span + 1 < h: grid[cx][ry + rh_span + 1] = 1
            for j in range(rh_span + 3):
                cy = ry - 1 + j
                if 0 <= cy < h:
                    if rx - 1 >= 0: grid[rx - 1][cy] = 1
                    if rx + rw_span + 1 < w: grid[rx + rw_span + 1][cy] = 1
            
            for i in range(rw_span + 1):
                for j in range(rh_span + 1):
                    grid[rx + i][ry + j] = 0
            
            boss_locations.append((rx + rw_span // 2, ry + rh_span // 2))
            
            door_x, door_y = random.choice(candidates)
            grid[door_x][door_y] = 0
                
    return boss_locations

# ==========================================
# 4. MAIN OPERATOR
# ==========================================
class MESH_OT_generate_maze(bpy.types.Operator):
    bl_idname = "mesh.generate_maze"
    bl_label = "Generate Maze"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.maze_props
        
        if props.use_random_seed: props.seed = random.randint(0, 999999)
        random.seed(props.seed)
        
        algo = props.maze_algorithm
        w = props.maze_width if props.maze_width % 2 != 0 else props.maze_width + 1
        h = props.maze_height if props.maze_height % 2 != 0 else props.maze_height + 1
        
        wall_h = props.wall_height
        size = props.cell_size
        thick_ratio = props.wall_thickness
        jitter = props.wall_jitter
        
        # --- CLEANUP ---
        if "Maze" in bpy.data.collections:
            collection = bpy.data.collections["Maze"]
            for obj in collection.objects: bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(collection)

        maze_col = bpy.data.collections.new("Maze")
        context.scene.collection.children.link(maze_col)
        
        # --- MATERIALS ---
        wall_mat_name = "WallMat"
        if props.use_brick_wall:
            wall_mat = create_brick_material(wall_mat_name, props.brick_scale)
        else:
            wall_mat = bpy.data.materials.get(wall_mat_name)
            if not wall_mat:
                wall_mat = bpy.data.materials.new(wall_mat_name)
            wall_mat.use_nodes = False
            wall_mat.diffuse_color = (0.1, 0.1, 0.12, 1.0)

        # --- GRID ---
        grid = [[1 for _ in range(h)] for _ in range(w)]

        # --- RUN ALGO ---
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        sx, sy = 1, 1
        if algo == 'BACKTRACKER': algo_recursive_backtracker(grid, w, h, sx, sy, directions)
        elif algo == 'HUNT_KILL': algo_hunt_and_kill(grid, w, h, sx, sy, directions)
        elif algo == 'PRIMS': algo_prims(grid, w, h, sx, sy, directions)
        elif algo == 'KRUSKAL': algo_kruskal(grid, w, h)
        elif algo == 'SIDEWINDER': algo_sidewinder(grid, w, h)
        elif algo == 'ELLER': algo_eller(grid, w, h)
        elif algo == 'RECURSIVE_DIVISION': algo_recursive_division(grid, w, h)

        # --- GATES ---
        grid[1][0] = 0
        grid[w-2][h-1] = 0
        solver_start = (1, 0)
        solver_end = (w-2, h-1)
        
        # --- POST PROCESS ---
        if props.braid_chance > 0: apply_braiding(grid, w, h, props.braid_chance)
        
        boss_spots = []
        if props.room_count > 0:
            protected = solve_bfs_path(grid, w, h, solver_start, solver_end)
            boss_spots = apply_rooms_smart(grid, w, h, props.room_count, props.room_size_min, props.room_size_max, protected)

        objects_to_unwrap = []

        # --- FLOOR ---
        floor_verts = [(0, 0, 0), (w * size, 0, 0), (w * size, h * size, 0), (0, h * size, 0)]
        floor_faces = [(0, 1, 2, 3)]
        floor_mesh = bpy.data.meshes.new("MazeFloorMesh")
        floor_mesh.from_pydata(floor_verts, [], floor_faces)
        floor_obj = bpy.data.objects.new("MazeFloor", floor_mesh)
        floor_obj.data.materials.append(wall_mat) 
        maze_col.objects.link(floor_obj)
        objects_to_unwrap.append(floor_obj)
        
        # --- CEILING ---
        if props.add_ceiling:
            ceil_verts = [(0, 0, wall_h), (w * size, 0, wall_h), (w * size, h * size, wall_h), (0, h * size, wall_h)]
            ceil_faces = [(3, 2, 1, 0)]
            ceil_mesh = bpy.data.meshes.new("MazeCeilingMesh")
            ceil_mesh.from_pydata(ceil_verts, [], ceil_faces)
            ceil_obj = bpy.data.objects.new("MazeCeiling", ceil_mesh)
            ceil_obj.data.materials.append(wall_mat)
            maze_col.objects.link(ceil_obj)
            objects_to_unwrap.append(ceil_obj)
        
        # --- WALLS ---
        verts, faces = [], []
        vert_index = 0
        
        def add_wall(x, y):
            nonlocal vert_index
            cx, cy = x * size + size / 2, y * size + size / 2
            half_thick = (size * thick_ratio) / 2
            x_min, x_max = cx - half_thick, cx + half_thick
            y_min, y_max = cy - half_thick, cy + half_thick
            if x > 0 and grid[x-1][y] == 1: x_min = x * size
            if x < w - 1 and grid[x+1][y] == 1: x_max = (x + 1) * size
            if y > 0 and grid[x][y-1] == 1: y_min = y * size
            if y < h - 1 and grid[x][y+1] == 1: y_max = (y + 1) * size
            
            current_h = wall_h
            if jitter > 0:
                current_h += random.uniform(-jitter, jitter)
                if current_h < 0.2: current_h = 0.2

            v = [
                (x_min, y_min, 0), (x_max, y_min, 0), (x_max, y_max, 0), (x_min, y_max, 0),
                (x_min, y_min, current_h), (x_max, y_min, current_h), (x_max, y_max, current_h), (x_min, y_max, current_h)
            ]
            verts.extend(v)
            f = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            faces.extend([tuple(i + vert_index for i in face) for face in f])
            vert_index += 8

        for x in range(w):
            for y in range(h):
                if grid[x][y] == 1: add_wall(x, y)

        mesh = bpy.data.meshes.new("MazeMesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        if hasattr(mesh, "use_auto_smooth"): mesh.use_auto_smooth = True
        obj = bpy.data.objects.new("MazeObject", mesh)
        maze_col.objects.link(obj)
        obj.data.materials.append(wall_mat)
        
        if props.auto_uv: objects_to_unwrap.append(obj)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        
        # --- UV UNWRAP ---
        if props.auto_uv and objects_to_unwrap:
            for target_obj in objects_to_unwrap:
                bpy.ops.object.select_all(action='DESELECT')
                target_obj.select_set(True)
                context.view_layer.objects.active = target_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.cube_project(cube_size=size * 2, correct_aspect=True)
                bpy.ops.object.mode_set(mode='OBJECT')
        
        # --- LIGHTS (THE MISSING LOGIC RESTORED!) ---
        if props.add_lights:
            light_data = bpy.data.lights.new(name="MazeTorch", type='POINT')
            light_data.energy = props.light_power
            light_data.color = props.light_color
            light_data.shadow_soft_size = 0.5
            
            light_spots = []
            for x in range(1, w-1, 2):
                for y in range(1, h-1, 2):
                    if grid[x][y] == 0:
                        neighbors = 0
                        if grid[x+1][y] == 0: neighbors += 1
                        if grid[x-1][y] == 0: neighbors += 1
                        if grid[x][y+1] == 0: neighbors += 1
                        if grid[x][y-1] == 0: neighbors += 1
                        
                        # Place lights at corners/dead ends
                        if neighbors != 2 and random.random() < 0.4:
                            light_spots.append((x, y))
            
            for (lx, ly) in light_spots:
                light_obj = bpy.data.objects.new(name="Torch", object_data=light_data)
                light_obj.location = (lx*size + size/2, ly*size + size/2, wall_h * 0.7)
                maze_col.objects.link(light_obj)

        # --- MARKERS ---
        if props.show_markers:
            def create_marker(lx, ly, is_start=True):
                m_verts, m_faces = [], []
                r, h_marker, segments = size * 0.3, wall_h * 1.5, 16
                m_verts.append((0, 0, 0))
                for i in range(segments):
                    angle = (i / segments) * 2 * math.pi
                    m_verts.append((math.cos(angle)*r, math.sin(angle)*r, 0))
                m_verts.append((0, 0, h_marker))
                for i in range(segments):
                    angle = (i / segments) * 2 * math.pi
                    m_verts.append((math.cos(angle)*r, math.sin(angle)*r, h_marker))
                for i in range(segments):
                    next_i = (i + 1) % segments
                    m_faces.append((1+i, 1+next_i, segments+2+next_i, segments+2+i))
                wx, wy = lx * size + size / 2, ly * size + size / 2
                m_verts_world = [(vx + wx, vy + wy, vz) for vx, vy, vz in m_verts]
                name = "StartMarker" if is_start else "EndMarker"
                mesh = bpy.data.meshes.new(name)
                mesh.from_pydata(m_verts_world, [], m_faces)
                obj = bpy.data.objects.new(name, mesh)
                maze_col.objects.link(obj)
                mat_name = "MarkerStart" if is_start else "MarkerEnd"
                mat = bpy.data.materials.get(mat_name)
                if not mat:
                    mat = bpy.data.materials.new(mat_name)
                    mat.use_nodes = False
                    mat.diffuse_color = (0.0, 1.0, 0.2, 1.0) if is_start else (1.0, 0.0, 0.2, 1.0)
                obj.data.materials.append(mat)
            create_marker(solver_start[0], solver_start[1], True)
            create_marker(solver_end[0], solver_end[1], False)

        # --- LOOT ---
        if props.loot_count > 0:
            potential_spots = []
            for x in range(w):
                for y in range(h):
                    if grid[x][y] == 0:
                        if (x, y) != solver_start and (x, y) != solver_end:
                            potential_spots.append((x, y))
            count = min(props.loot_count, len(potential_spots))
            loot_spots = random.sample(potential_spots, count) if potential_spots else []
            if loot_spots:
                l_verts = [(0,0,0.4), (0,0,-0.4), (0.3,0,0), (0,0.3,0), (-0.3,0,0), (0,-0.3,0)]
                l_faces = [(0,2,3), (0,3,4), (0,4,5), (0,5,2), (1,3,2), (1,4,3), (1,5,4), (1,2,5)]
                l_verts = [(vx*size/1.5, vy*size/1.5, vz*size/1.5) for vx,vy,vz in l_verts]
                loot_mesh = bpy.data.meshes.new("LootMesh")
                loot_mesh.from_pydata(l_verts, [], l_faces)
                mat_name = "LootGold"
                l_mat = bpy.data.materials.get(mat_name)
                if not l_mat:
                    l_mat = bpy.data.materials.new(mat_name)
                    l_mat.use_nodes = False
                    l_mat.diffuse_color = (1.0, 0.84, 0.0, 1.0)
                loot_mesh.materials.append(l_mat)
                for (lx, ly) in loot_spots:
                    wx, wy = lx*size + size/2, ly*size + size/2
                    wz = wall_h / 2.0
                    loot_obj = bpy.data.objects.new("Loot", loot_mesh)
                    loot_obj.location = (wx, wy, wz)
                    maze_col.objects.link(loot_obj)
                    if props.add_physics:
                        context.view_layer.objects.active = loot_obj
                        bpy.ops.rigidbody.object_add()
                        loot_obj.rigid_body.type = 'ACTIVE'
                        loot_obj.rigid_body.collision_shape = 'CONVEX_HULL'
        
        # --- BOSSES ---
        if boss_spots:
            b_verts = [(0, 0, wall_h * 0.8), (-0.4, -0.4, 0), (0.4, -0.4, 0), (0.4, 0.4, 0), (-0.4, 0.4, 0)]
            b_faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 3, 2)]
            b_verts = [(vx * size, vy * size, vz) for vx, vy, vz in b_verts]
            boss_mesh = bpy.data.meshes.new("BossMesh")
            boss_mesh.from_pydata(b_verts, [], b_faces)
            mat_name = "BossMat"
            b_mat = bpy.data.materials.get(mat_name)
            if not b_mat:
                b_mat = bpy.data.materials.new(mat_name)
                b_mat.use_nodes = False
                b_mat.diffuse_color = (0.8, 0.0, 0.0, 1.0)
            boss_mesh.materials.append(b_mat)
            for (bx, by) in boss_spots:
                wx, wy = bx * size + size / 2, by * size + size / 2
                boss_obj = bpy.data.objects.new("Boss", boss_mesh)
                boss_obj.location = (wx, wy, 0)
                maze_col.objects.link(boss_obj)
                if props.add_physics:
                    context.view_layer.objects.active = boss_obj
                    bpy.ops.rigidbody.object_add()
                    boss_obj.rigid_body.type = 'ACTIVE'
                    boss_obj.rigid_body.mass = 5.0
                    boss_obj.rigid_body.collision_shape = 'CONVEX_HULL'

        # --- PHYSICS ---
        if props.add_physics:
            if not context.scene.rigidbody_world: bpy.ops.rigidbody.world_add()
            context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add()
            obj.rigid_body.type = 'PASSIVE'
            obj.rigid_body.collision_shape = 'MESH'
            context.view_layer.objects.active = floor_obj
            bpy.ops.rigidbody.object_add()
            floor_obj.rigid_body.type = 'PASSIVE'
            floor_obj.rigid_body.collision_shape = 'BOX'

        # --- SOLVER ---
        if props.show_solution:
            path_set = solve_bfs_path(grid, w, h, solver_start, solver_end)
            def solve_bfs_ordered(grid, start, end):
                queue = [[start]]
                visited = {start}
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                while queue:
                    path = queue.pop(0)
                    node = path[-1]
                    if node == end: return path
                    cx, cy = node
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if grid[nx][ny] == 0 and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                new_path = list(path)
                                new_path.append((nx, ny))
                                queue.append(new_path)
                return []
            final_path = solve_bfs_ordered(grid, solver_start, solver_end)
            if final_path:
                sol_verts = [(p[0]*size + size/2, p[1]*size + size/2, wall_h/2) for p in final_path]
                sol_edges = [(i, i+1) for i in range(len(sol_verts)-1)]
                sol_mesh = bpy.data.meshes.new("SolutionMesh")
                sol_mesh.from_pydata(sol_verts, sol_edges, [])
                sol_obj = bpy.data.objects.new("SolutionPath", sol_mesh)
                maze_col.objects.link(sol_obj)
                mat_name = "SolutionMat"
                mat = bpy.data.materials.get(mat_name)
                if not mat:
                    mat = bpy.data.materials.new(mat_name)
                    mat.use_nodes = False
                    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
                sol_obj.data.materials.append(mat)

        return {'FINISHED'}

# ==========================================
# 5. UI & REGISTER
# ==========================================
class MAZE_PT_main_panel(bpy.types.Panel):
    bl_label = "Maze Generator"
    bl_idname = "MAZE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Maze Pro" 

    def draw(self, context):
        layout = self.layout
        props = context.scene.maze_props

        # Section 1: Core
        box = layout.box()
        box.label(text="Core Settings", icon='PREFERENCES')
        box.prop(props, "maze_algorithm", text="")
        row = box.row()
        row.prop(props, "maze_width")
        row.prop(props, "maze_height")
        box.prop(props, "cell_size")
        
        # Section 2: Styling
        box = layout.box()
        box.label(text="Geometry & Style", icon='MOD_BUILD')
        box.prop(props, "wall_height")
        box.prop(props, "wall_thickness", slider=True)
        box.prop(props, "wall_jitter", slider=True)
        
        row = box.row()
        row.prop(props, "use_brick_wall")
        if props.use_brick_wall:
            row.prop(props, "brick_scale")
            
        row = box.row()
        row.prop(props, "add_ceiling")
        row.prop(props, "auto_uv")

        # Section 3: Gameplay
        box = layout.box()
        box.label(text="Gameplay & Layout", icon='OBJECT_DATA')
        box.prop(props, "braid_chance", slider=True)
        box.prop(props, "room_count")
        row = box.row(align=True)
        row.prop(props, "room_size_min")
        row.prop(props, "room_size_max")
        layout.separator()
        box.prop(props, "loot_count")
        box.prop(props, "show_markers")
        box.prop(props, "show_solution")
        
        # Section 4: Atmosphere & Physics
        box = layout.box()
        box.label(text="Atmosphere & Physics", icon='WORLD_DATA')
        box.prop(props, "add_lights")
        if props.add_lights:
            row = box.row()
            row.prop(props, "light_color", text="")
            row.prop(props, "light_power")
        box.prop(props, "add_physics")
        
        # Section 5: Seed
        box = layout.box()
        box.prop(props, "use_random_seed")
        row = box.row()
        row.prop(props, "seed")
        row.enabled = not props.use_random_seed
        
        layout.separator()
        row = layout.row()
        row.scale_y = 2.0
        row.operator("mesh.generate_maze", icon='MESH_GRID')

classes = (MazeProperties, MESH_OT_generate_maze, MAZE_PT_main_panel)

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.maze_props = bpy.props.PointerProperty(type=MazeProperties)

def unregister():
    for cls in classes: bpy.utils.unregister_class(cls)
    del bpy.types.Scene.maze_props

if __name__ == "__main__":
    register()