import itertools

import numpy as np


def jolts_part1(bank):
    result = 0
    for i, batt1 in enumerate(bank[:-1]):
        for j, batt2 in enumerate(bank[i + 1:]):
            result = max(result, batt1 * 10 + batt2)
    
    return result


def jolts_part2(bank, select):
    result = []
    for n in range(select):
        if -select + n + 1 != 0:
            temp = bank[:-select + n + 1]
        else:
            temp = bank
        
        maxloc = np.argmax(temp)
        result.append(temp[maxloc])
        bank = bank[maxloc + 1:]
    
    result_int = 0
    for i, num in enumerate(result):
        result_int += (10 ** (select - i - 1)) * num
    
    return result_int


def main():
    with open("input.txt") as f:
        file = f.read().splitlines()
    
    batteries = [
        list(map(int, list(line)))
        for line in file
    ]
    
    print(jolts_part2([1, 2, 3], 3))
    
    part1 = sum(jolts_part1(battery) for battery in batteries)
    print(f"Part 1: {part1}")
    
    part2 = sum(jolts_part2(battery, 12) for battery in batteries)
    print(f"Part 2: {part2}")


if __name__ == "__main__":
    main()
