

res = {}


def w(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return 1
    if a > 20 or b > 20 or c > 20:
        return w(20, 20, 20)
    if (a, b, c) in res:
        return res[(a, b, c)]
    if a < b < c:
            res[(a, b, c)] = w(a, b, c - 1) + w(a, b - 1, c - 1) - w(a, b - 1, c)
    else:
            res[(a, b, c)] = w(a - 1, b, c) + w(a - 1, b - 1, c) + w(a - 1, b, c - 1) - w(a - 1, b - 1, c - 1)
    return res[(a, b, c)]

while True:
    a, b, c = map(int, input().split())
    if a == b == c == -1:
        print()
        break
    if (a, b, c) not in res.keys():
        temp = w(a, b, c)
        res[(a, b, c)] = temp
        print(f'w({a}, {b}, {c}) = {temp}')
    else:
        print(f'w({a}, {b}, {c}) = {res[(a, b, c)]}')