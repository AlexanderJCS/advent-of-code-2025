from functools import lru_cache


def parse():
    with open("input.txt") as f:
        lines = f.read().splitlines()
    
    return {
        line.split(": ")[0]: line.split(": ")[1].split(" ")
        for line in lines
    }


def connections_to_out(connections: dict, current="you"):
    total = 0
    for connection in connections[current]:
        if connection == "out":
            return 1
        total += connections_to_out(connections, connection)
    
    return total


def connections_to_out_visiting(connections: dict, start="svr"):
    @lru_cache(maxsize=8192)
    def dfs(current="svr", visited_dac=False, visited_fft=False):
        if current == "out":
            return 1 if visited_dac and visited_fft else 0
        
        if current == "dac":
            visited_dac = True
        if current == "fft":
            visited_fft = True
        
        return sum(
            dfs(connection, visited_dac, visited_fft)
            for connection in connections[current]
        )
    
    return dfs(start)


def main():
    connections = parse()
    
    print(f"Part 1: {connections_to_out(connections)}")
    print(f"Part 2: {connections_to_out_visiting(connections)}")


if __name__ == "__main__":
    main()
