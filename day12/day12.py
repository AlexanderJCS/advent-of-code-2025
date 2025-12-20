import math


def parse():
    with open("input.txt") as f:
        lines = f.read().splitlines()
    
    present_areas = []
    for line in lines:
        if not line:
            continue
        
        if line[1] == ":":
            present_areas.append(0)
            continue
        
        if len(line) > 5 and line[5] == ":":
            break
        
        present_areas[-1] += line.count("#")
    
    region_reqs = []
    
    for line in lines:
        if len(line) <= 5 or line[5] != ":":
            continue
        
        a, b = line.split(": ")
        area = math.prod(tuple(map(int, a.split("x"))))
        present_counts = tuple(map(int, b.split(" ")))
        
        region_reqs.append((area, present_counts))
    
    return present_areas, region_reqs


def main():
    present_areas, region_reqs = parse()
    
    result = 0
    for region_req in region_reqs:
        region_present_areas = 0
        for present_idx, present_count in enumerate(region_req[1]):
            region_present_areas += present_areas[present_idx] * present_count
        
        if region_present_areas <= region_req[0]:
            result += 1
    
    print(f"Part 1: {result}")


if __name__ == "__main__":
    main()
