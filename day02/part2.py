

def read(filename):
    with open(filename, "r") as f:
        return f.read().strip().replace("\n", "").split(",")


def is_repeating(num):
    for i in range(1, len(num)//2 + 1):
        if num[:i] * (len(num) // i) == num:
            return True
    
    return False


def main():
    ranges = read("input.txt")
    
    answer = 0
    for r in ranges:
        start, end = map(int, r.split("-"))
        for num in range(start, end + 1):
            if is_repeating(str(num)):
                answer += num
    
    print(answer)


if __name__ == "__main__":
    main()
