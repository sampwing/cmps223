from decorators import memoize

@memoize
def factor(n):
    return {item for item in range(1, n + 1) if n % item == 0}

def factors(x):
    return {key: value for (key, value) in 
            zip(range(1, x + 1), 
            map(factor, range(1, x + 1)))}

@memoize
def totient(x):
    factored = factors(x)
    match_set = factor(x)
    relatively_prime = 0
    for key, value in factored.iteritems():
        if match_set.intersection(value) == {1}:
            relatively_prime += 1
    return relatively_prime 


def fs(x):
    t1 = totient(x)
    pairs = []
    for i in xrange(1, t1 + 1):
        ti = totient(i)
        for j in xrange(i, t1 + 1):
            tj = totient(j)
            if ti * tj == t1:
                pairs.append([i, j])
    return pairs

if __name__ == '__main__':
    #print totient(7031)
    import json
    json.dump(fs(7031), open('pairs.output', 'wb'))

