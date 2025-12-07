

def union(ranges_tuples):
    # Version 2 listed in: https://stackoverflow.com/a/15273749/18758660
    result = []
    for begin, end in sorted(ranges_tuples):
        if result and result[-1][1] >= begin - 1:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    
    return result


def main():
    with open("input.txt") as f:
        data = f.read().splitlines()
    
    ranges = []
    fresh = 0
    
    for line in data:
        if not line:
            continue
        
        if "-" in line:
            start, end = map(int, line.split("-"))
            ranges.append(range(start, end + 1))
            continue
        
        num = int(line)
        for r in ranges:
            if num in r:
                fresh += 1
                break
    
    print(f"Part 1: {fresh}")
    
    ranges_tuples = [(r.start, r.stop - 1) for r in ranges]
    unioned = union(ranges_tuples)
    
    fresh_ids = 0
    for r in unioned:
        fresh_ids += r[1] - r[0] + 1
    
    print(f"Part 2: {fresh_ids}")
    
        

if __name__ == "__main__":
    main()
