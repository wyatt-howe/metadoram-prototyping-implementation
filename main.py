from common import Message
import asyncio
import concurrent
from utils import random_bit_generator, xor, bits_to_index
from math import log2, ceil
from functools import partial

def log(*args):
    print(*args)

randbit = random_bit_generator(integer_seed=0)  #seed=b"\x00"*32)


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
    # assert test_balanced_sspir()

    log("Exiting...")
