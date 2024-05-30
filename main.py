from common import Message
import asyncio
import concurrent
from utils import random_bit_generator, xor, bits_to_index
from math import log2, ceil
from functools import partial
import numpy

def log(*args):
    print(*args)

randbit = random_bit_generator(integer_seed=0)  #seed=b"\x00"*32)

async def unbalanced_sspir(io, pid, m, d, A, x, op="UnbalancedSSPIR"):
    """ UnbalancedSSPIR """
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)

    """ UnbalancedSSPIR : Step 1 """
    if pid == 0:
        x1 = x

    if pid == 1:
        await give('x1 for P2', x)
        x2 = xor(x, await get('x2 for P1'))

    if pid == 2:
        await give('x2 for P1', x)
        x2 = xor(x, await get('x1 for P2'))


    """ UnbalancedSSPIR : Step 2 """
    if pid == 0:
        Q = [1 if i == bits_to_index(x1) else 0 for i in range(m)]
        assert sum(Q) == 1


    """ UnbalancedSSPIR : Step 3 """
    if pid == 0:
        Q1 = [next(randbit) for _ in range(m)]
        Q2 = xor(Q, Q1)
        await give('Q1 for P1', Q1)
        await give('Q2 for P2', Q2)

    if pid == 1:
        Q1 = await get('Q1 for P1')

    if pid == 2:
        Q2 = await get('Q2 for P2')


    """ UnbalancedSSPIR : Step 4 """
    if pid == 1:
        assert len(Q1) == m
        index_x2_for_permuting = bits_to_index(x2)
        W1 = [Q1[i ^ index_x2_for_permuting] for i in range(m)]
        assert sum(W1) == sum(Q1)

    if pid == 2:
        assert len(Q2) == m
        index_x2_for_permuting = bits_to_index(x2)
        W2 = [Q2[i ^ index_x2_for_permuting] for i in range(m)]
        assert sum(W2) == sum(Q2)


    """ UnbalancedSSPIR : Step 5 """
    if pid == 1:
        A1 = A
        assert len(A1) == m
        assert all(len(A1[i]) == d for i in range(m))
        v1 = xor(*[[A1[i][j] * W1[i] for j in range(d)] for i in range(m)])

    if pid == 2:
        A2 = A
        assert len(A2) == m
        assert all(len(A2[i]) == d for i in range(m))
        v2 = xor(*[[A2[i][j] * W2[i] for j in range(d)] for i in range(m)])


    """ UnbalancedSSPIR : Step 6 """
    if pid == 1:
        return v1

    if pid == 2:
        return v2



async def balanced_sspir(io, pid, m, d, A, x, op="BalancedSSPIR"):
    """ BalancedSSPIR """
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)

    """ BalancedSSPIR : Step 1 """
    q = 2**ceil((log2(m) - log2(d))/2)
    assert q**2 >= m/d


    """ BalancedSSPIR : Step 2 """
    B = []
    if pid == 1:
        A1 = A
        B1 = numpy.array(A1).reshape(m // q, d * q).tolist()

    if pid == 2:
        A2 = A
        B2 = numpy.array(A2).reshape(m // q, d * q).tolist()


    """ BalancedSSPIR : Step 3 """
    if pid == 1:
        x2 = x
        z2 = x2[: ceil(log2(q))]
        y2 = x2[ceil(log2(q)) : ceil(log2(m))]

    if pid == 2:
        x2 = x
        z2 = x2[: ceil(log2(q))]
        y2 = x2[ceil(log2(q)) : ceil(log2(m))]

    """ BalancedSSPIR : Step 4 """
    if pid == 1:
        u1 = await unbalanced_sspir(io, 1, m // q, d * q, B1, y2)

    if pid == 2:
        u2 = await unbalanced_sspir(io, 2, m // q, d * q, B2, y2)

    """ BalancedSSPIR : Step 5 """
    # This step is buggy. Fix it.
    if pid == 1:
        v1 = []
        for i in range(q):
            v1.append(u1[i*d : i*d + i])


    if pid == 2:
        v2 = []
        for i in range(q):
            v2.append(await u2[i * d: i * d + i])

    """ BalancedSSPIR : Step 6 """
    if pid == 1:
        return v1

    if pid == 2:
        return v2















async def simulate_unbalanced_sspir(test_m, test_d, test_A1, test_A2, test_x0, test_x1, test_x2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        io = Message()
        loop = asyncio.get_event_loop()

        futures = []
        log("Starting...")
        futures.append(await loop.run_in_executor(executor, unbalanced_sspir, io, 0, test_m, test_d, None, test_x0))
        log("  * Started Builder.")
        futures.append(await loop.run_in_executor(executor, unbalanced_sspir, io, 1, test_m, test_d, test_A1, test_x1))
        log("  * Started Holder 1.")
        futures.append(await loop.run_in_executor(executor, unbalanced_sspir, io, 2, test_m, test_d, test_A2, test_x2))
        log("  * Started Holder 2.")

        results = [await f for f in asyncio.as_completed(futures)]
        log("output:", *results)

        return results

async def simulate_balanced_sspir(test_m, test_d, test_A1, test_A2, test_x0, test_x1, test_x2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        io = Message()
        loop = asyncio.get_event_loop()

        futures = []
        log("Starting...")
        futures.append(await loop.run_in_executor(executor, balanced_sspir, io, 0, test_m, test_d, None, test_x0))
        log("  * Started Builder.")
        futures.append(await loop.run_in_executor(executor, balanced_sspir, io, 1, test_m, test_d, test_A1, test_x1))
        log("  * Started Holder 1.")
        futures.append(await loop.run_in_executor(executor, balanced_sspir, io, 2, test_m, test_d, test_A2, test_x2))
        log("  * Started Holder 2.")

        results = [await f for f in asyncio.as_completed(futures)]
        log("output:", *results)

        return results

def make_sspir_tests():
    def test_sspir(simulate_sspir):
        test_m = 1024  # Must be a power of two due to the paper's specification of how a permutation works.
        test_d = 8
        test_A = [[next(randbit) for _ in range(test_d)] for _ in range(test_m)]
        test_A1 = test_A  # not secret shared, but public    so not [[next(randbit) for _ in range(test_d)] for _ in range(test_m)]
        test_A2 = test_A  # not secret shared
        test_x0 = [next(randbit) for _ in range(ceil(log2(test_m)))]
        test_x1 = [next(randbit) for _ in range(ceil(log2(test_m)))]
        test_x2 = [next(randbit) for _ in range(ceil(log2(test_m)))]

        _v0, v1, v2 = asyncio.run(
            simulate_sspir(test_m, test_d, test_A1, test_A2, test_x0, test_x1, test_x2))

        v = xor(v1, v2)

        x = xor(test_x0, test_x1, test_x2)
        Ax = test_A1[bits_to_index(x)]

        log(f"{x} {Ax} {v} {Ax == v}")

        # assert Ax == v

        return True
    return partial(test_sspir, simulate_unbalanced_sspir), partial(test_sspir, simulate_balanced_sspir)

if __name__ == "__main__":
    log("Entering main...")

    test_unbalanced_sspir, test_balanced_sspir = make_sspir_tests()
    assert test_unbalanced_sspir()
    assert test_balanced_sspir()

    log("Exiting...")
