def f(x):
    def factor(n):
        arr = set()
        for i in xrange(1, n + 1):
            if n % i == 0: arr.add(i)
        return arr
    match_set = factor(x)
    factored = {}
    for i in xrange(1, x):
        factored[i] = factor(i)
    relatively_prime = 0
    for key, value in factored.iteritems():
        print key, match_set, value
        if match_set.intersection(value) == {1}:
            relatively_prime += 1
    return relatively_prime 

if __name__ == '__main__':
    #print f(7031)
    print f(9)
    print f(1)
