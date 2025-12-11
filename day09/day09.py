import numpy as np


def main():
    with open("input.txt") as f:
        coords_txt = f.read().splitlines()
    
    coords = []
    for coord_txt in coords_txt:
        coords.append(list(map(int, coord_txt.split(","))))
    
    coords = np.array(coords)
    print(coords.min(axis=0), coords.max(axis=0))


if __name__ == "__main__":
    main()
