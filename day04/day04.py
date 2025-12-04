

def load():
    with open("input.txt") as f:
        lines = f.read().splitlines()
    
    return list(map(list, lines))


def get_at(level, x, y):
    if x < 0 or x >= len(level[0]) or y < 0 or y >= len(level):
        return "."
    return level[y][x]


def adjacent_paper_rolls(level, x, y):
    paper_rolls = 0
    for r in range(x - 1, x + 2):
        for c in range(y - 1, y + 2):
            if r == x and c == y:
                continue
            
            if get_at(level, r, c) == "@":
                paper_rolls += 1
    
    return paper_rolls


def remove_paper_rolls(level):
    accessible = 0
    
    pos_to_remove = []
    
    for r in range(len(level[0])):
        for c in range(len(level)):
            if level[r][c] != "@":
                continue
            
            if adjacent_paper_rolls(level, c, r) < 4:
                accessible += 1
                pos_to_remove.append((r, c))
    
    for r, c in pos_to_remove:
        level[r][c] = "."
    
    return accessible


def print_level(level):
    for row in level:
        for ch in row:
            print(ch, end="")
        print()


def main():
    level = load()
    
    part_1 = remove_paper_rolls(level)
    print(f"Part 1: {part_1}")
    
    part_2 = part_1
    while (removed := remove_paper_rolls(level)) > 0:
        part_2 += removed
    
    print(f"Part 2: {part_2}")


if __name__ == "__main__":
    main()
