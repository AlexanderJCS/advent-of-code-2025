from dataclasses import dataclass
from collections import deque

import numpy as np
from tqdm import tqdm


@dataclass
class Machine:
    lights: list[bool]
    schematics: list[tuple[int, ...]]
    joltage_req: list[int]


def findall(s, find):
    return [i for i, c in enumerate(s) if c == find]


def parse_line(line):
    lights_str = line[1:line.find("]")]
    lights = [ch == "#" for ch in lights_str]
    
    joltage_str = line[line.find("{") + 1:line.find("}")]
    joltages = list(map(int, joltage_str.split(",")))
    
    schematic_starts = findall(line, "(")
    schematic_ends = findall(line, ")")
    
    schematics = []
    for start, end in zip(schematic_starts, schematic_ends):
        schematics.append(tuple(map(int, line[start + 1:end].split(","))))
    
    return Machine(
        lights,
        schematics,
        joltages
    )


def parse():
    with open("input.txt") as f:
        lines = f.read().splitlines()
    
    return [parse_line(line) for line in lines]


def execute_instruction(lights, flip_locs):
    new = [light for light in lights]  # deep copy
    for loc in flip_locs:
        new[loc] = not new[loc]
    
    return new


def turn_on_lights(machine: Machine):
    queue = deque()
    visited = set()

    target = tuple(machine.lights)

    for schematic in machine.schematics:
        queue.append(([False for _ in range(len(target))], schematic, 1))

    while queue:
        lights, schematic, instruction_no = queue.popleft()
        new_lights = execute_instruction(list(lights), schematic)
        new_state = tuple(new_lights)

        if new_state in visited:
            continue
        visited.add(new_state)

        if new_state == target:
            return instruction_no

        for new_schematic in machine.schematics:
            if new_schematic == schematic:
                continue
            
            queue.append((new_state, new_schematic, instruction_no + 1))

    raise ValueError("All-on state is unreachable")


def get_to_joltage(machine: Machine):
    queue = deque()
    visited = set()
    
    target = np.array(machine.joltage_req)
    schematics_np = [np.array(s) for s in machine.schematics]
    for schematic in schematics_np:
        queue.append((np.array([0 for _ in range(len(target))]), schematic, 1))
    
    while queue:
        joltage, schematic, presses = queue.popleft()
        new_joltage = joltage.copy()
        new_joltage[schematic] += 1
        new_joltage_t = tuple(new_joltage)
        
        if new_joltage_t in visited:
            continue
        visited.add(new_joltage_t)
        
        if np.any(new_joltage > target):
            continue
        
        if np.all(new_joltage == target):
            print(new_joltage)
            return presses
        
        for new_schematic in schematics_np:
            queue.append((new_joltage, new_schematic, presses + 1))
    
    raise ValueError("Target joltage is unreachable")


def main():
    machines = parse()

    part_1 = sum(turn_on_lights(machine) for machine in machines)
    print(f"Part 1: {part_1}")
    
    part_2 = sum(get_to_joltage(machine) for machine in tqdm(machines))
    print(f"Part 2: {part_2}")
    

if __name__ == "__main__":
    main()
