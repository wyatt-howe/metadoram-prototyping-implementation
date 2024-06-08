import math

from common import Message
import asyncio
import concurrent
from utils import random_bit_generator, xor, bits_to_index, permutation
from math import log2, ceil
from functools import partial
import numpy
import random
def log(*args):
    print(*args)

randbit = random_bit_generator(integer_seed=0)  #seed=b"\x00"*32)


#
# Implementation of some helpers needed in the below figures.
#

def xor_bit(b1, b2):
    return b1 ^ b2

def and_bit(b1, b2):
    return b1 & b2

def and_secret_bit_start(xi, yi, abci, i_equals_j):
    ai, bi, _ = abci
    di = xor_bit(xi, ai)
    ei = xor_bit(yi, bi)
    return (di, ei), (abci, i_equals_j)

def and_secret_bit_end(di, ei, dni, eni, prev_abci, i_equals_j):
    d = xor_bit(di, dni)
    e = xor_bit(ei, eni)
    ai, bi, ci = prev_abci
    zi = xor_bit(ci, xor_bit(and_bit(ai, e), and_bit(bi, d)))
    if i_equals_j:
        zi = xor_bit(zi, and_bit(d, e))
    return zi

async def and_with_jit_triple_generation(io, pid, xi, yi, op="MultiplyJIT"):
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)

    if pid == 0:
        # Generate a triple
        aa = next(randbit)
        bb = next(randbit)
        cc = aa & bb

        as1 = next(randbit)
        as2 = aa ^ as1

        bs1 = next(randbit)
        bs2 = bb ^ bs1

        cs1 = next(randbit)
        cs2 = cc ^ cs1

        await give('a1,b1,c1 for P1', (as1, bs1, cs1))
        await give('a2,b2,c2 for P2', (as2, bs2, cs2))

    if pid == 1:
        triple_a1_b1_c1 = await get('a1,b1,c1 for P1')
        de1, state1 = and_secret_bit_start(xi, yi, triple_a1_b1_c1, pid==2)
        await give('de1 for P2', de1)  # Server 1 reveals its d1 and e1 shares

    if pid == 2:
        triple_a2_b2_c2 = await get('a2,b2,c2 for P2')
        de2, state2 = and_secret_bit_start(xi, yi, triple_a2_b2_c2, pid==2)
        await give('de2 for P1', de2)  # Server 2 reveals its d1 and e1 shares

    if pid == 1:
        __de2 = await get('de2 for P1')
        return and_secret_bit_end(*de1, *__de2, *state1)

    if pid == 2:
        __de1 = await get('de1 for P2')
        return and_secret_bit_end(*de2, *__de1, *state2)


#
# Implementation of figures in the paper.
#

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
    if pid == 0:
        x0 = x
        z0 = x0[: ceil(log2(q))]
        y0 = x0[ceil(log2(q)) : ceil(log2(m))]

    if pid == 1:
        x1 = x
        z1 = x1[: ceil(log2(q))]
        y1 = x1[ceil(log2(q)) : ceil(log2(m))]

    if pid == 2:
        x2 = x
        z2 = x2[: ceil(log2(q))]
        y2 = x2[ceil(log2(q)) : ceil(log2(m))]

    """ BalancedSSPIR : Step 4 """
    if pid == 0:
        u1 = await unbalanced_sspir(io, 0, m // q, d * q, None, y0, op="BalancedSSPIR_subroutine_step_4")

    if pid == 1:
        u1 = await unbalanced_sspir(io, 1, m // q, d * q, B1, y1, op="BalancedSSPIR_subroutine_step_4")

    if pid == 2:
        u2 = await unbalanced_sspir(io, 2, m // q, d * q, B2, y2, op="BalancedSSPIR_subroutine_step_4")

    """ BalancedSSPIR : Step 5 """
    # This step is buggy...
    if pid == 1:
        v1 = []
        for i in range(q):
            v1.append(u1[i*d : i*d + i])

    if pid == 2:
        v2 = []
        for i in range(q):
            v2.append(u2[i * d: i * d + i])

    """ BalancedSSPIR : Step 6 """
    if pid == 1:
        return v1

    if pid == 2:
        return v2

#
# Implementation of ReadWrite - Figure 1
#
async def ReadWrite(io, op, pid, x, y, f):
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)
    # ReadWrite : Step 1
    v = Read(x)
    # ReadWrite : Step 2
    v_new = f.read(v, y)
    # ReadWrite : Step 3
    n = await get("n")
    t = await get("t")
    d = await get("d")
    c = (log2(n) * log2(n) / log2(log2(n)))
    await Write(io, op, pid, n, t, d, x, v_new)
    # ReadWrite : Step 4
    await Rebuild(io, op, pid, n, c, t, d)
    # ReadWrite : Step 5
    t = t + 1
    await give("t", t)
    # ReadWrite : Step 6
    return v

#
# Implementation of Write - Figure 1
#
async def Write(io, op, pid, n, t, d, x, v_new):
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)

    if pid == 0:
        c = (log2(n)*log2(n)/log2(log2(n)))
        # Write: Step 1
        r = random.choice(range(1, 2 * n + 1))

        # Write: Step 2
        j = t % c

        # Write : Step 3
        e = ""
        for _ in range(d):
            e+=""+str(random.choice([0,1]))

        await give("e", e)

        # Write : Step 5
        V = await get('V')
        V[0][j] = v_new
        await give('V', V)

        R = await get('R')
        R[0][j] = r
        await give('R', R)

        X = await get('X')
        X[0][j] = x
        await give('X', X)

    if pid == 1:
        e = get("e")
        # Write : Step 4
        C1 = []
        for j in range(len(v_new)):
            C1[j] = xor_bit(v_new[j], e[j])

    if pid == 2:
        e = get("e")
        # Write : Step 4
        C1 = []
        for j in range(len(v_new)):
            C1[j] = xor_bit(v_new[j], e[j])

#
# Implementation of Rebuild. Figure 1
#
async def Rebuild(io, op, pid, n, c, t, d):
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)

    # Rebuild: Step 1
    l = log2(n/c)*log2(n)*log2(n)
    for i in range(l):
        bi_c = (math.pow(log2(n), 0.5*i))*c
        # Rebuild: Step 1 (a)
        if t == (0%bi_c):
            # Rebuild : Step 1 (a) (i)
            biplus_c = (math.pow(log2(n), 0.5 * (i+1)))*c
            u = (t/bi_c)%biplus_c
            # Rebuild : Step 1 (a) (ii)
            for j in range(bi_c):
                V = await get('V')
                V[i+1][u*bi_c + j ] = V[i][j]
                V[i][j] = None
                await give('V', V)

                X = await get('X')
                X[i+1][u*bi_c + j ] =X[i][j]
                X[i][j] = None
                await give('X', X)

            if pid == 0:

                R = await get('R')
                R[i + 1][u * bi_c + j] = R[i][j]
                R[i][j] = None
                await give('R', R)

                e = ""
                for _ in range(d):
                   e+=""+str(random.choice([0,1]))
                E = await get('E')
                E[i + 1][u * bi_c + j] = e
                await give('E', E)

                Z = await get('Z')
                Z[i + 1][u * bi_c + j] = xor(V[i+1][u*bi_c + j ], E[i + 1][u * bi_c + j])
                await give('Z', Z)

                # Rebuild : Step (a) (iii). Missing.
                # P0 must build a Let [Q]0 be the injective mapping from [b
                # i
                # c] to [2(1+ϵ)b
                # i
                # c]
                # that maps Ri,1...bic
                # to satisfying locations with these hash functions.

                # Rebuild : Step (a) (iv).
                give("Q0", [])

            if pid == 1:
                T = await get('T')
                Z = await get('Z')
                Q0 = await get('Q0')
                T[i + 1][u] = fRoute(Z[i+1], Q0)
                await give('T', T)

            if pid == 2:
                T = await get('T')
                Z = await get('Z')
                Q0 = await get('Q0')
                T[i + 1][u] = fRoute(Z[i + 1], Q0)
                await give('T', T)
        if t == n:
            Refresh()

#
# Implementation of init. Figure 2
#
async def init(io, op, pid, n, V):
    # Init : Step 1
    if pid == 0:
        M_0 = permutation(2*n)
        R_0 = []
        B = [0]* n-1
        for i in range(1,n):
            R_0[i] = M_0[i]
            if i<n/2:
                B[i] = M_0[2*i - 1].append(M_0[2*i])
            subDoram = await fDoram_Init(io, op, n/2, 2(log2(n)+1), B)
    pass

#
# Implementation of fDoram_init.
#
async def fDoram_Init(io, op, n, d, A):
    give = partial(io.give, op_id=op)
    give("A", A)
    give("n", n)
    give("d", d)
    return A

def fRoute(Z, Q):
    return []
    pass

#
# Implementation of Extract - Figure 3
#
def Extract(io, op, pid):
    give = partial(io.give, op_id=op)
    get = partial(io.get, op_id=op)

    # Extract : Step 1
    R = get("R")
    R_concatenate = []
    for _ in R:
        R_concatenate.extend(_)

    V = get("V")
    V_concatenate = []
    for _ in R:
        V_concatenate.extend(_)

    X = get("X")
    X_concatenate = []
    for _ in X:
        X_concatenate.extend(_)

    # Extract : Step 2
    m = len(X_concatenate)
    D = get('D')

    # Extract : Step 3
    if pid == 1:
        S_1 = permutation(m)
        R_1 = fRoute(R, S_1)
        V_1 = fRoute(V, S_1)
        X_1 = fRoute(X, S_1)

        # Extract : Step 5
        R = get("R")

        # Extract : Step 6

        D_1 = D
        I = [0]*(m-1)
        for i in range(1, m):
            if R_1[i] not in D_1:
                I[i] = 1

            else:
                X_1[i] = None
                V_1[i] = None
        X = get("X")

        # Extract : Step 7
        for i in range(V_1):
            temp = V[i]
            V_1[i] = V_1[X[i]]
            V_1[X[i]] = temp

        # Extract : Step 8 and 9
        return V_1

    if pid == 2:
        U_2 = permutation(m)
        R_2 = fRoute(R, U_2)
        V_2 = fRoute(V, U_2)
        X_2 = fRoute(X, U_2)

        # Extract : Step 5
        R = get("R")
        # Extract : Step 6
        D_2 = D
        I = [0] * (m - 1)
        for i in range(1, m):
            if R_2[i] not in D_2:
                I[i] = 1
            else:
                X_2[i] = None
                V_2[i] = None
        X = get("X")

        # Extract : Step 7
        for i in range(V_2):
            temp = V[i]
            V_2[i] = V_2[X[i]]
            V_2[X[i]] = temp

        # Extract : Step 8 and 9
        return V_2


#

#
# Implementation for Refresh method. Figure 3
#
def Refresh(io, op, pid):
    V = Extract(io, op, pid)
    init(0, 0, V)

#
# Methods for testing a run of the above code.
#

async def simulate_unbalanced_sspir(test_m, test_d, test_A1, test_A2, test_x0, test_x1, test_x2):
    log("Starting...")
    io, loop, executor = Message(), asyncio.get_event_loop(), concurrent.futures.ThreadPoolExecutor()
    results = [await f for f in asyncio.as_completed([
        await loop.run_in_executor(executor, unbalanced_sspir, io, 0, test_m, test_d, None,    test_x0),
        await loop.run_in_executor(executor, unbalanced_sspir, io, 1, test_m, test_d, test_A1, test_x1),
        await loop.run_in_executor(executor, unbalanced_sspir, io, 2, test_m, test_d, test_A2, test_x2)
    ])]
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
        Axbig = test_A1[bits_to_index(x, bitorder='big')]
        Axlit = test_A1[bits_to_index(x, bitorder='little')]

        log(f"x={x}\nA[x]_lit={Axlit}\nA[x]_big={Axbig}\nv={v}\npassed={any([Axlit == v, Axbig == v])}\nbits_to_index(x)={bits_to_index(x)}")

        return any((Axlit == v, Axbig == v))
    return partial(test_sspir, simulate_unbalanced_sspir), partial(test_sspir, simulate_balanced_sspir)

async def simulate_jit_and(x1, y1, x2, y2):
    io, loop, executor = Message(), asyncio.get_event_loop(), concurrent.futures.ThreadPoolExecutor()
    _, z1, z2 = [await f for f in asyncio.as_completed([
        await loop.run_in_executor(executor, and_with_jit_triple_generation, io, 0, None, None),
        await loop.run_in_executor(executor, and_with_jit_triple_generation, io, 1, x1, y1),
        await loop.run_in_executor(executor, and_with_jit_triple_generation, io, 2, x2, y2)
    ])]
    return z1, z2

def test_jit_and():
    results = []
    for x in (0, 1):
        for y in (0, 1):
            for r_x in (0, 1):
                for r_y in (0, 1):
                    x1, x2 = r_x, xor_bit(r_x, x)
                    y1, y2 = r_y, xor_bit(r_y, y)
                    z1, z2 = asyncio.run(simulate_jit_and(x1, y1, x2, y2))
                    z = xor_bit(z1, z2)
                    results.append(and_bit(x, y) == z)
    return all(results)


if __name__ == "__main__":
    log("Entering main...")

    test_unbalanced_sspir, test_balanced_sspir = make_sspir_tests()
    # assert test_unbalanced_sspir()
    # assert test_balanced_sspir()

    assert test_jit_and()

    log("Exiting...")
