import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]



class MyClass():
    def __init__(self) -> None:
        pass


    
    def is_prime(self, n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        sqrt_n = int(math.floor(math.sqrt(n)))
        for i in range(3, sqrt_n + 1, 2):
            if n % i == 0:
                return False
        return True

    def foo(self, x,y):
        return x+y
        

    

bar = [1, 2, 3]
baz = [4, 5, 6]

def foo_(self, x,y):
        return x+y


def main():
    cls = MyClass()
    with concurrent.futures.ProcessPoolExecutor() as executor:
            for n1, n2, prime in zip(bar, baz, executor.map(cls.foo, bar, baz)):
                print('%d + %d = %s' % (n1, n2, prime))

    import dill
    print(dill.pickles(foo_))
    

if __name__ == '__main__':
    main()
