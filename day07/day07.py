

def find(s, k):
    assert(len(k) == 1)
    
    found = []
    for i, c in enumerate(s):
        if c == k:
            found.append(i)
    
    return found


def split_beams(beam_locations: set[int], beam_splitters: set[int], universes: dict[int, int]):
    num_beams_before = len(beam_locations)
    
    split_count = 0
    for beam_splitter in beam_splitters:
        if beam_splitter not in beam_locations:
            continue
        
        universes[beam_splitter - 1] = universes.get(beam_splitter - 1, 0) + universes[beam_splitter]
        universes[beam_splitter + 1] = universes.get(beam_splitter + 1, 0) + universes[beam_splitter]
        universes[beam_splitter] = 0
        
        beam_locations.remove(beam_splitter)
        beam_locations.add(beam_splitter - 1)
        beam_locations.add(beam_splitter + 1)
        
        split_count += 1
    
    return split_count, len(beam_locations) - num_beams_before


def main():
    with open("input.txt") as f:
        rows = f.read().splitlines()
    
    split_times = 0
    timelines = 0
    beams = set(find(rows[0], "S"))
    universes = {beam: 1 for beam in beams}  # there's only 1 beam, so we can take this shortcut
    for row in rows[1:]:
        splitters = set(find(row, "^"))
        new_split, new_timelines = split_beams(beams, splitters, universes)
        split_times += new_split
        timelines += new_timelines
    
    print(f"Part 1: {split_times}")
    print(f"Part 2: {sum(universes.values())}")


if __name__ == "__main__":
    main()
