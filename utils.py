import pyaes

def random_byte_generator(seed):
    key = (seed)
    enc = pyaes.AESModeOfOperationCTR(key).encrypt
    ctr = 0
    while True:
        pt = int.to_bytes(ctr, length=32, byteorder='big')
        bs = enc(pt)
        for byte in bs:
            yield byte
        ctr = ctr + 1

def random_bit_generator(integer_seed):
    seed = int.to_bytes(integer_seed, length=32, byteorder='big')
    randbytes = random_byte_generator(seed)
    for byte in randbytes:
        for bit_idx in range(8):
            yield (byte >> bit_idx) & 1

def xor(*args):
    if len(args) > 0:
        xs0 = args[0]
        for xsi in args[1:]:
            for j, xsij in enumerate(xsi):
                xs0[j] = xs0[j] ^ xsij
        return xs0

def bits_to_index(bits, bitorder='big'):
    bits_ = bits if bitorder=='little'\
        else (
            reversed(bits) if bitorder=='big'
            else
                ValueError('Invalid endianness specified.')
        )
    return sum([2**i * b for i, b in enumerate(bits_)])
